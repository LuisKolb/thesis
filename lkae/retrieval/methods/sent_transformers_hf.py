from typing import List
import requests
import os

from lkae.retrieval.types import EvidenceRetriever, EvidenceRetrieverResult
from lkae.utils.data_loading import AuthorityPost

import logging
logger = logging.getLogger(__name__)


class HFSentenceTransformersRetriever(EvidenceRetriever):
    """
    sentence-transformers via 🤗 Inference API (remotely)
    see also: https://huggingface.co/docs/inference-endpoints/index
    """

    def __init__(
        self,
        retriever_k,
        model="sentence-transformers/multi-qa-distilbert-cos-v1",
        api_key="",
        **kwargs,
    ):
        print(f"Initializing HFSentenceTransformersRetriever with model: {model}")

        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.API_URL = f"https://api-inference.huggingface.co/models/{model}"

        super().__init__(retriever_k)

    def retrieve(
        self, rumor_id: str, claim: str, timeline: List[AuthorityPost], **kwargs
    ) -> List[EvidenceRetrieverResult]:

        tweets: List[str] = [t.text for t in timeline]
        tweet_ids: List[str] = [t.post_id for t in timeline]

        # models need to load...

        similarity_scores = self.query(
            {"inputs": {"source_sentence": claim, "sentences": tweets}}
        )

        # sort the timeline by similarity score
        sorted_timeline = sorted(
            zip(tweet_ids, tweets, similarity_scores), key=lambda x: x[2], reverse=True
        )

        ranked = []

        for i, ranked_tweet in enumerate(sorted_timeline[: self.k]):
            ranked.append(
                EvidenceRetrieverResult(rumor_id, ranked_tweet[0], i+1, ranked_tweet[2])
            )

        return ranked

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[0]
    retriever = HFSentenceTransformersRetriever(5)
    data = retriever.retrieve(
        rumor_id=sample["id"], claim=sample["rumor"], timeline=sample["timeline"]
    )

    for d in data:
        print(d)
