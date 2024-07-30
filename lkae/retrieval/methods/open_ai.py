import os
import numpy as np
from openai import OpenAI

# from sklearn.metrics.pairwise import cosine_similarity
from lkae.retrieval.retrieve import EvidenceRetriever

import logging
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), # This is the default and can be omitted
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class OpenAIRetriever(EvidenceRetriever):
    def __init__(self, k, api_key=None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        logger.info(f'OpenAI client initialized with {"API key from .env" if api_key else "provided API key"}')
        super().__init__(k)

    def get_embedding(self, text):
        response = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def get_embedding_multiple(self, texts):
        response = self.client.embeddings.create(
            input=texts, model="text-embedding-3-small"
        )
        return [r.embedding for r in response.data]

    def retrieve(self, rumor_id, claim, timeline, **kwargs):
        logger.info(f"retrieving documents for rumor_id: {rumor_id}")

        # Generate embedding for the claim
        claim_embedding = self.get_embedding(claim)

        # Generate embeddings for each entry in the timeline
        timeline_embeddings = self.get_embedding_multiple(
            [tweet[2] for tweet in timeline]
        )

        # Compute similarities
        similarities = [
            cosine_similarity(claim_embedding, tweet_embedding)
            for tweet_embedding in timeline_embeddings
        ]

        # Select the top k most relevant tweets based on similarities
        most_relevant_tweet_indices = np.argsort(similarities)[-self.k:][::-1]

        scores = [similarities[i] for i in most_relevant_tweet_indices]
        relevant_tweets = [timeline[i] for i in most_relevant_tweet_indices]

        ranked_results = []
        for i, (cos_sim, tweet) in enumerate(zip(scores, relevant_tweets)):
            ranked_results.append([rumor_id, tweet[1], i + 1, cos_sim])

        return ranked_results