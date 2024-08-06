from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from lkae.retrieval.retrieve import EvidenceRetriever
from lkae.retrieval.types import EvidenceRetrieverResult


class TFIDFRetriever(EvidenceRetriever):
    def __init__(self, retriever_k, **kwargs):
        super().__init__(retriever_k)

    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs) -> List[EvidenceRetrieverResult]:
        # Get only doc texts
        documents = [tweet[2] for tweet in timeline]
        tweet_ids = [tweet[1] for tweet in timeline]

        # Combine query and documents for TF-IDF vectorization
        combined_texts = [claim] + documents

        # Generate TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_texts)

        # Calculate similarity of the query to each document
        # use sklearn's cosine_similarity function to calculate pairwise cos similarity
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]) # type: ignore

        # Rank documents based on similarity scores
        ranked_doc_indices = similarity_scores.argsort()[0][::-1]

        ranked = []
        for i, idx in enumerate(ranked_doc_indices[:self.k]):
            ranked.append([rumor_id, tweet_ids[idx], i+1, similarity_scores[0][idx]])

        return ranked
    

if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[0]
    retriever = TFIDFRetriever(5)
    data = retriever.retrieve(
        rumor_id=sample["id"], claim=sample["rumor"], timeline=sample["timeline"]
    )

    for d in data:
        print(d)