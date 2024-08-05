from typing import List
from lkae.retrieval.retrieve import EvidenceRetriever


# simple implementation of BM25 for ranking documents based on a query
# from https://gist.github.com/koreyou/f3a8a0470d32aa56b32f198f49a9f2b8

""" Implementation of OKapi BM25 with sklearn's TfidfVectorizer
Distributed as CC-0 (https://creativecommons.org/publicdomain/zero/1.0/)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


class BM25(object):
    def __init__(self, b=0.75, k1=1.6):
        self.vectorizer = TfidfVectorizer(norm=None, smooth_idf=False)
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = y.sum(1).mean()

    def transform(self, q, X):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        X = super(TfidfVectorizer, self.vectorizer).transform(X)
        len_X = X.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = X.tocsc()[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        # idf(t) = log [ n / df(t) ] + 1 in sklearn, so it need to be coneverted
        # to idf(t) = log [ n / df(t) ] with minus 1
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)                                                          
        return (numer / denom).sum(1).A1


class BM25Retriever(EvidenceRetriever):
    def __init__(self, retriever_k, **kwargs):
        super().__init__(retriever_k)

    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs):
        # Get only doc texts
        documents = [tweet[2] for tweet in timeline]
        tweet_ids = [tweet[1] for tweet in timeline]

        # Generate TF-IDF vectors
        bm25 = BM25()
        bm25.fit(documents)
        bm25_scores = bm25.transform(claim, documents)

        # Rank documents based on BM25 scores (reverse the order)
        ranked_doc_indices = bm25_scores.argsort()[::-1]

        ranked = []
        for i, idx in enumerate(ranked_doc_indices[:self.k]):
            ranked.append([rumor_id, tweet_ids[idx], i, bm25_scores[idx]])

        return ranked
    
    
if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl
    ds = load_pkl(os.path.join(pkl_dir, 'English_train', 'pre-nam-bio.pkl'))
    sample = ds[0]
    retriever = BM25Retriever(5)
    data = retriever.retrieve(rumor_id=sample['id'],
                              claim=sample['rumor'],
                              timeline=sample['timeline'])

    print(data)