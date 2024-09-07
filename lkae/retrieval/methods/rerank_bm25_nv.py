import os
import numpy as np
from typing import List
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from lkae.retrieval.types import EvidenceRetriever, EvidenceRetrieverResult
from lkae.retrieval.methods.bm25 import BM25
from lkae.utils.scoring import cosine_similarity

import logging
logger = logging.getLogger(__name__)


class RerankingRetriever(EvidenceRetriever):

    def __init__(
            self, retriever_k, b=0.75, k1=1.6, rerank_cutoff=50, retriever_model="nvidia/nv-embed-v1", api_key=None, **kwargs
    ):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.bm25_b = b
        self.bm25_k1 = k1
        
        self.rerank_cutoff = rerank_cutoff

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


    def retrieve(
        self, rumor_id: str, claim: str, timeline: List, **kwargs
    ) -> List[EvidenceRetrieverResult]:
        logger.info(f"retrieving documents for rumor_id: {rumor_id}")
        
        # Get only doc texts
        documents = [tweet[2] for tweet in timeline]

        # Generate TF-IDF vectors
        bm25 = BM25(b=self.bm25_b, k1=self.bm25_k1)
        bm25.fit(documents)
        bm25_scores = bm25.transform(claim, documents)

        # Rank documents based on BM25 scores (reverse the order)
        ranked_doc_indices = bm25_scores.argsort()[::-1]
        
        timeline_ranked = []

        # new timeline by ranked_doc_indices and cutoff
        for idx in ranked_doc_indices[: self.rerank_cutoff]: 
            timeline_ranked.append(timeline[idx])

        # Generate embedding for the claim
        claim_embedding = self.get_embedding_query(claim)

        # Generate embeddings for each entry in the timeline
        timeline_embeddings = self.get_embedding_passage(
            [tweet[2] for tweet in timeline_ranked],
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
        relevant_tweets = [timeline_ranked[i] for i in most_relevant_tweet_indices]

        reranked_results = []
        for i, (cos_sim, tweet) in enumerate(zip(scores, relevant_tweets)):
            reranked_results.append(
                EvidenceRetrieverResult(rumor_id, tweet[1], i + 1, cos_sim)
            )
        return reranked_results


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[3]
    retriever = RerankingRetriever(5)#, retriever_model="nvidia/nv-embedqa-e5-v5")
    data = retriever.retrieve(
        rumor_id=sample["id"], claim=sample["rumor"], timeline=sample["timeline"]
    )

    for d in data:
        print(d)