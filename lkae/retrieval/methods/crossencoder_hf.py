import time
from typing import Dict, List
import requests
import os

from lkae.retrieval.types import EvidenceRetriever, EvidenceRetrieverResult
from lkae.utils.data_loading import AuthorityPost

import logging
logger = logging.getLogger(__name__)

"""
unused
"""


class HFCrossencoderRetriever(EvidenceRetriever):
    """
    crossencoder via 🤗 Inference API (remotely)
    see also: https://huggingface.co/docs/inference-endpoints/index
    """

    def __init__(
        self,
        retriever_k,
        retriever_model="abbasgolestani/ag-nli-DeTS-sentence-similarity-v4",
        api_key="",
        **kwargs,
    ):
        self.model = retriever_model
        
        print(
            f"Initializing HFCrossencoderRetriever with model: {self.model}"
        )

        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.API_URL = f"https://api-inference.huggingface.co/models/{self.model}"

        self.query(
            self.format_input("warm me up scotty", "bzzt")
        )

        super().__init__(retriever_k)

    def format_input(self, claim: str, tweet: str) -> Dict:
        return {"text": claim, "text_target": tweet}

    def retrieve(
        self, rumor_id: str, claim: str, timeline: List[AuthorityPost], **kwargs
    ) -> List[EvidenceRetrieverResult]:

        tweets: List[str] = [t.text for t in timeline]
        tweet_ids: List[str] = [t.post_id for t in timeline]

        similarity_scores = []

        for tweet in tweets:
            similarity_score_dict = self.query(
                self.format_input(claim, tweet)
            ) # return a dict in the form of {'label': 'LABEL_0', 'score': 0.08930326253175735}
            similarity_scores.append(similarity_score_dict[0]["score"])

        # sort the timeline by similarity score
        sorted_timeline = sorted(
            zip(tweet_ids, tweets, similarity_scores), key=lambda x: x[2], reverse=True
        )

        ranked = []

        for i, ranked_tweet in enumerate(sorted_timeline[: self.k]):
            ranked.append(
                EvidenceRetrieverResult(
                    rumor_id, ranked_tweet[0], i + 1, ranked_tweet[2]
                )
            )

        return ranked

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        if response.status_code != 200:
            if response.status_code == 503:
                # need to warm up the model, see https://huggingface.co/docs/api-inference/quicktour#model-loading-and-latency
                res = response.json()
                print(
                    f"Waiting for model to warm up (for {float(res['estimated_time'])} seconds)"
                )
                time.sleep(float(res["estimated_time"]))
                response = requests.post(
                    self.API_URL, headers=self.headers, json=payload
                )
            else:
                raise ValueError(
                    f"Error: {response.status_code}; Text: {response.text}"
                )
        return response.json()


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[0]

    retriever = HFCrossencoderRetriever(
        5, retriever_model="abbasgolestani/ag-nli-DeTS-sentence-similarity-v4"
    )
    data = retriever.retrieve(
        rumor_id=sample["id"], claim=sample["rumor"], timeline=sample["timeline"]
    )
    for d in data:
        print(d)
