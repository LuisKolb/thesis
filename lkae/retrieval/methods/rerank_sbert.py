from typing import List
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from lkae.retrieval.types import EvidenceRetriever, EvidenceRetrieverResult

import logging
logger = logging.getLogger(__name__)


class CrossEncoderRerankRetriever(EvidenceRetriever):
    """
    sentence-transformers via sentence_transformers Python library (locally)
    see also: https://huggingface.co/docs/inference-endpoints/index
    """

    def __init__(self, 
                 retriever_k,
                 rerank_cutoff=20,
                 embedding_model="sentence-transformers/msmarco-distilbert-cos-v5", 
                 reranking_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        ):
        self.embedder = SentenceTransformer(embedding_model)
        self.cross_encoder = CrossEncoder(reranking_model, default_activation_function=torch.nn.Sigmoid())

        self.rerank_cutoff = rerank_cutoff

        super().__init__(k=retriever_k)

    def retrieve(
        self, rumor_id: str, claim: str, timeline: List, **kwargs
    ) -> List[EvidenceRetrieverResult]:
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        corpus = [t[2] for t in timeline]
        top_k = min(self.k, len(corpus))

        query_embedding = self.embedder.encode(claim, device=device)
        corpus_embeddings = self.embedder.encode(corpus, device=device)

        similarities = self.embedder.similarity(query_embedding, corpus_embeddings) # type: ignore

        cos_scores = similarities.data.tolist()[0]
        
        ranked_timeline = []
        for tweet, score in zip(timeline, cos_scores):
            ranked_timeline.append(tweet)
        ranked_timeline = ranked_timeline[: self.rerank_cutoff]

        passages = [t[2] for t in ranked_timeline]

        ranks = self.cross_encoder.rank(claim, 
                                        passages, 
                                        top_k=top_k, 
                                        return_documents=False)
        
        ranked_results = []
        for i, rank in enumerate(ranks):
            ranked_results.append(
                EvidenceRetrieverResult(rumor_id, ranked_timeline[rank["corpus_id"]][1], i+1, rank['score']) # type: ignore
            )

        return ranked_results


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[0]
    retriever = CrossEncoderRerankRetriever(5)
    data = retriever.retrieve(
        rumor_id=sample["id"], claim=sample["rumor"], timeline=sample["timeline"]
    )

    for d in data:
        print(d)
