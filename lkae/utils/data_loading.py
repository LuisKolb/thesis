from typing import Generator, List, TypedDict, Union, Optional, NamedTuple, Dict
import json
import os
import re
import pickle as pkl

import lkae
from lkae.utils.preprocessing import clean_text_custom

import logging

logger = logging.getLogger(__name__)


class AuthorityPost(NamedTuple):
    """
    url: str
    post_id: str
    text: str
    rank: Optional[int]
    score: Optional[float]
    """

    url: str
    post_id: str
    text: str
    rank: Optional[int]
    score: Optional[float]


class RumorWithEvidence(TypedDict):
    """
    id: str
    rumor: str
    label: Optional[str]
    timeline: List[AuthorityPost]
    evidence: Optional[List[AuthorityPost]] # not required
    retrieved_evidence: Optional[List[AuthorityPost]] # not required
    """

    id: str
    rumor: str
    label: Optional[str]
    timeline: List[AuthorityPost]
    evidence: Optional[List[AuthorityPost]]  # not required
    retrieved_evidence: Optional[List[AuthorityPost]]  # not required


# paths in the modules
root_dir = os.path.dirname(lkae.__file__)  # module root
pkl_dir = os.path.join(root_dir, "index")

eng_combined_jsonl = os.path.join(root_dir, "data", "English_combined.jsonl")
eng_dev_jsonl = os.path.join(root_dir, "data", "English_dev.jsonl")
eng_train_jsonl = os.path.join(root_dir, "data", "English_train.jsonl")


class AuredDataset(object):
    def __init__(
        self,
        filepath,
        preprocess,
        add_author_name,
        add_author_bio,
        # blind_run=True,
        blacklisted_rumor_ids=[],
        author_info_filepath="../../lkae/combined-author-data-translated.json",
        **kwargs,
    ) -> None:
        self.filepath: Union[str, os.PathLike] = filepath
        self.rumors: List[RumorWithEvidence] = []

        """
        init ds like this (for example):

        config = {
            'preprocess': True,
            'add_author_name': False,
            'add_author_bio': False,
            ...
        }
        ds = AuredDataset(filepath, **config)
        """
        self.preprocess: bool = preprocess
        self.add_author_name: bool = add_author_name
        self.add_author_bio: bool = add_author_bio
        self.author_info_filepath: str = author_info_filepath
        self.blacklisted_rumor_ids: List[str] = blacklisted_rumor_ids

        self.load_rumor_data()

    def __str__(self) -> str:
        return json.dumps(self.rumors, indent=2)

    def __iter__(self) -> Generator[RumorWithEvidence, None, None]:
        for rumor_item in self.rumors:
            yield rumor_item

    def __getitem__(self, idx):
        return self.rumors[idx]

    def __setitem__(self, idx, val):
        self.rumors[idx] = val

    def __len__(self) -> int:
        return len(self.rumors)

    def load_rumor_data(self):
        jsons = self.load_rumors_from_jsonl()

        for item in jsons:
            entry = RumorWithEvidence(item)
            if self.blacklisted_rumor_ids and entry["id"] in self.blacklisted_rumor_ids:
                continue # skip blacklisted rumors
            entry["timeline"] = [AuthorityPost(*post, None, None) for post in entry["timeline"]]  # type: ignore
            if "evidence" in entry and entry["evidence"]:
                entry["evidence"] = [AuthorityPost(*post, None, None) for post in entry["evidence"]]  # type: ignore
            entry["retrieved_evidence"] = []
            self.rumors.append(entry)

        logger.info(f"loaded {len(jsons)} json entries from {self.filepath}")

        for item in self.rumors:
            item["timeline"] = self.format_posts(item["timeline"])
            if "evidence" in item and item["evidence"]:
                item["evidence"] = self.format_posts(item["evidence"])
            if self.preprocess:
                item["rumor"] = clean_text_custom(item["rumor"])

    def get_grouped_rumors(self):
        """
        returns a dict with mapping {rumor_id: RumorWithEvidence}
        """
        grouped = {}
        for item in self.rumors:
            grouped[item["id"]] = item  # add to grouped dict
        return grouped

    def load_rumors_from_jsonl(self) -> List[RumorWithEvidence]:
        jsons = []
        with open(self.filepath, encoding="utf-8") as file:
            for line in file:
                jsons += [json.loads(line)]
        return jsons

    def format_posts(self, post_list: List[AuthorityPost]):
        new_post_list = []
        author_info = {}
        if self.add_author_bio or self.add_author_name:
            with open(self.author_info_filepath, "r") as file:
                author_info = json.load(file)

        for post in post_list:
            # use regex to verify if the account url is valid
            if (
                re.match(r"(https:\/\/)?twitter.com/[\w+]{1,15}\b", post.url)
                and not self.add_author_name
            ):
                new_post_text = f'Statement from Authority Account "{post.url.split("/")[-1]}": "{post.text}"'
            else:
                # dont add author handle here if author name is added later
                new_post_text = f'Statement: "{post.text}"'

            if author_info:
                account = post.url.strip()
                if not account.startswith("https://"):
                    account = f"https://{account}"
                name = author_info[account]["translated_name"]
                bio = author_info[account]["translated_bio"]

                if bio and self.add_author_bio:
                    new_post_text = f'Authority Description: "{bio}"\n' + new_post_text
                if name and self.add_author_name:
                    new_post_text = f'Authority Name: "{name}"\n' + new_post_text

                # if all info is added, evidence will look like this:
                #
                # Authority Description: "{bio}"
                # Authority Name: "{name}"
                # Statement from {twitter handle}: "{post.text}"

            if self.preprocess:
                new_post_text = clean_text_custom(new_post_text)

            new_post_list.append(
                AuthorityPost(post.url, post.post_id, new_post_text, None, None)
            )
        return new_post_list

    def add_trec_file_judgements(
        self, trec_judgements_path, sep=" ", normalize_scores=True
    ):
        """
        create a list of RankedDocs objects in key retrieved_evidence from TREC-formatted file

        Parameters:
            - trec_judgements_path: filepath to TREC-formatted file containing rank and score
            - sep: separator used in TREC file

        Returns:
        list of json-like rumors from dataset, with the key retrieved_evidence populated with a list of RankedDocs-like objects
        """
        trec_by_id = {}
        max_score = 0
        min_score = 0
        num_rows = 0

        with open(trec_judgements_path, "r") as file:
            for line in file:
                # handle edge case where dataset may contain field which contain an additional whitespace, ...
                # which was then saved together with the field value as a string looking like 'id 0 "id " 1'
                # (only seems to affect TERRIER trec files)
                line = re.sub('"', "", line)
                line = re.sub("  ", " ", line)
                rumor_id, _, evidence_id, rank, score, tag = line.split(sep)
                score = float(score)
                if rumor_id not in trec_by_id:
                    trec_by_id[rumor_id] = {}

                # keep max,min score values to normalize later
                # scores should always be positive, but still do this...
                if score < min_score:
                    min_score = score
                if score > max_score:
                    max_score = score

                # add entry to dict for lookup later
                trec_by_id[rumor_id][evidence_id] = (rank, score)
                num_rows += 1

        if (max_score - min_score) == 0:
            logger.error(
                f"encountered (max_score-min_score) == 0; max={max_score}; min={min_score}"
            )
            raise ValueError()

        for i, item in enumerate(self.rumors):
            timeline: List[AuthorityPost] = item["timeline"]

            item["retrieved_evidence"] = []

            doc_ranks = trec_by_id[item["id"]]

            for post in timeline:
                if post.post_id in doc_ranks:
                    rank, score = doc_ranks[post.post_id]

                    # normalize score to [0...1] using max,min scores from earlier
                    if not normalize_scores:
                        score_norm = score
                    else:
                        score_norm = (score - min_score) / (max_score - min_score)

                    item["retrieved_evidence"].append(
                        AuthorityPost(
                            post.url,
                            post.post_id,
                            post.text,
                            int(rank),
                            float(score_norm),
                        )
                    )

            self.rumors[i] = item

        logger.info(
            f"added {num_rows} scores from {trec_judgements_path} to the evidence entries"
        )

    def add_trec_list_judgements(self, trec_judgements_list, normalize_scores=True):
        """
        create a list of RankedDocs objects in key retrieved_evidence from TREC-formatted list

        Parameters:
            - trec_judgements_list: TREC-formatted list containing rank and score
            - sep: separator used in TREC file

        Returns:
        none, modifies the dataset in place
        """
        trec_by_id = {}
        max_score = 0
        min_score = 0
        num_rows = 0

        for trec_judgements in trec_judgements_list:

            rumor_id, evidence_id, rank, score = trec_judgements
            rank += 1  # convert to 1-based ranking, is usually 0-based
            score = float(score)
            if rumor_id not in trec_by_id:
                trec_by_id[rumor_id] = {}

            # keep max,min score values to normalize later
            # scores should always be positive, but still do this...
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score

            # add entry to dict for lookup later
            trec_by_id[rumor_id][evidence_id] = (rank, score)
            num_rows += 1

        if (max_score - min_score) == 0:
            logger.error(
                f"encountered (max_score-min_score) == 0; max={max_score}; min={min_score}"
            )
            raise ValueError()

        for i, item in enumerate(self.rumors):
            timeline: List[AuthorityPost] = item["timeline"]

            item["retrieved_evidence"] = []

            doc_ranks = trec_by_id[item["id"]]

            for post in timeline:
                if post.post_id in doc_ranks:
                    rank, score = doc_ranks[post.post_id]

                    # normalize score to [0...1] using max,min scores from earlier
                    if not normalize_scores:
                        score_norm = score
                    else:
                        score_norm = (score - min_score) / (max_score - min_score)

                    item["retrieved_evidence"].append(
                        AuthorityPost(
                            post.url,
                            post.post_id,
                            post.text,
                            int(rank),
                            float(score_norm),
                        )
                    )

            self.rumors[i] = item

        logger.info(
            f"added {num_rows} scores from trec_judgements_list to the evidence entries"
        )


def load_pkl(file_path) -> AuredDataset:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file {file_path} does not exist")
    with open(file_path, "rb") as file:
        return pkl.load(file)


def load_pkls(index_dir: str = pkl_dir) -> Dict[str, Dict[str, AuredDataset]]:
    """
    load all pkl files in a directory and return a dict with mapping
    {dataset_split: dataset}
    """

    datasets = {}
    datasets = {}

    for subdir in os.listdir(index_dir):
        if not os.path.isdir(os.path.join(index_dir, subdir)):
            continue

        datasets[subdir] = {}

        for filename in os.listdir(os.path.join(index_dir, subdir)):
            if not filename.endswith('.pkl'):
                continue
            
            key = os.path.join(subdir, filename)
            datasets[subdir][filename.split('.')[0]] = load_pkl(os.path.join(index_dir, key))

    return datasets