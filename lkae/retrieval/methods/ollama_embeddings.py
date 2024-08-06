import numpy as np
from openai import OpenAI

from lkae.retrieval.types import EvidenceRetriever
from lkae.utils.scoring import cosine_similarity

import logging
logger = logging.getLogger(__name__)


class OllamaEmbeddingRetriever(EvidenceRetriever):
    def __init__(self, retriever_k, retriever_model="mxbai-embed-large", **kwargs):
        self.client = OpenAI(
            base_url="http://localhost:11435/v1",
            api_key="ollama",  # required, but unused
        )
        self.model = retriever_model
        logger.info(f"OpenAI client initialized with ollama")
        super().__init__(retriever_k)

    def get_embedding(self, text):
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def get_embedding_multiple(self, texts):
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [r.embedding for r in response.data]

    def retrieve(self, rumor_id, claim, timeline, **kwargs):
        logger.info(f"retrieving documents for rumor_id: {rumor_id}")

        # Generate embedding for the claim
        # claim_embedding = self.get_embedding(claim)

        # Generate embeddings for each entry in the timeline
        embeddings = self.get_embedding_multiple(
            [claim, *[tweet[2] for tweet in timeline]]
        )

        # Compute similarities
        claim_embedding = embeddings[0]
        timeline_embeddings = embeddings[1:]
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
            ranked_results.append([rumor_id, tweet[1], i+1, cos_sim])

        return ranked_results


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[0]
    retriever = OllamaEmbeddingRetriever(5)
    data = retriever.retrieve(
        rumor_id=sample["id"], claim=sample["rumor"], timeline=sample["timeline"]
    )

    for d in data:
        print(d)
