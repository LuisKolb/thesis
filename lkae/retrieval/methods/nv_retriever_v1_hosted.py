import os
import numpy as np
from openai import OpenAI

from lkae.retrieval.types import EvidenceRetriever
from lkae.utils.scoring import cosine_similarity

import logging
logger = logging.getLogger(__name__)

"""
UNUSED
"""


class NVRetrieverV1Hosted(EvidenceRetriever):
    def __init__(
        self, retriever_k, retriever_model="nvidia/nv-embed-v1", api_key=None
    ):
        self.client = OpenAI(api_key=api_key or os.environ.get("NV_API_KEY"),
                             base_url="https://integrate.api.nvidia.com/v1")
        
        logger.info(f'Client initialized with {"API key from .env" if api_key else "provided API key"}')

        self.model = retriever_model
        self.truncate = "END"

        super().__init__(retriever_k)

    def get_embedding_query(self, text):
        response = self.client.embeddings.create(input=text,
                                                 model=self.model,
                                                 encoding_format="float",
                                                 extra_body={"input_type": "query", "truncate": self.truncate})
        return response.data[0].embedding

    def get_embedding_passage(self, texts, rumor_id):
        response = self.client.embeddings.create(input=texts,
                                                 model=self.model,
                                                 encoding_format="float",
                                                 extra_body={"input_type": "passage", "truncate": self.truncate})
        if not response:
            raise ValueError(f"Error: no response from NV API for {rumor_id} with texts: {texts}")
        if type(response) is str:
            print(response)
            raise ValueError(f"Error: response from NV API for {rumor_id} is a string")
        return [r.embedding for r in response.data]

    def retrieve(self, rumor_id, claim, timeline, **kwargs):
        logger.info(f"retrieving documents for rumor_id: {rumor_id}")

        # Generate embedding for the claim
        claim_embedding = self.get_embedding_query(claim)

        # Generate embeddings for each entry in the timeline
        timeline_embeddings = self.get_embedding_passage(
            [tweet[2] for tweet in timeline],
            rumor_id
        )

        # Compute similarities
        similarities = [
            cosine_similarity(claim_embedding, tweet_embedding)
            for tweet_embedding in timeline_embeddings
        ]

        # Select the top k most relevant tweets based on similarities
        most_relevant_tweet_indices = np.argsort(similarities)[-self.k :][::-1]

        scores = [similarities[i] for i in most_relevant_tweet_indices]
        relevant_tweets = [timeline[i] for i in most_relevant_tweet_indices]

        ranked_results = []
        for i, (cos_sim, tweet) in enumerate(zip(scores, relevant_tweets)):
            ranked_results.append([rumor_id, tweet[1], i + 1, cos_sim])

        return ranked_results


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[3]
    retriever = NVRetrieverV1Hosted(5, retriever_model="nvidia/nv-embedqa-e5-v5")
    data = retriever.retrieve(
        rumor_id=sample["id"], claim=sample["rumor"], timeline=sample["timeline"][0:90]
    )

    for d in data:
        print(d)



