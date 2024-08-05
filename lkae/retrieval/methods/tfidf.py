from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lkae.retrieval.retrieve import EvidenceRetriever


class TFIDFRetriever(EvidenceRetriever):
    def __init__(self, retriever_k, **kwargs):
        super().__init__(retriever_k)

    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs):
        # Get only doc texts
        documents = [tweet[2] for tweet in timeline]
        tweet_ids = [tweet[1] for tweet in timeline]

        # Combine query and documents for TF-IDF vectorization
        combined_texts = [claim] + documents

        # Generate TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_texts)

        # Calculate similarity of the query to each document
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]) # type: ignore

        # Rank documents based on similarity scores
        ranked_doc_indices = similarity_scores.argsort()[0][::-1]

        ranked = []
        for i, idx in enumerate(ranked_doc_indices[:self.k]):
            ranked.append([rumor_id, tweet_ids[idx], i, similarity_scores[0][idx]])

        return ranked