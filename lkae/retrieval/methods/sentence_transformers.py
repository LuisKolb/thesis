from typing import List
from sentence_transformers import SentenceTransformer, util
import torch

from lkae.retrieval.retrieve import EvidenceRetriever

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class SBERTRetriever(EvidenceRetriever):
    def __init__(self, k, embedding_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        super().__init__(k)

    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs):
    
        corpus = [t[2] for t in timeline]
        corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)

        top_k = min(self.k, len(corpus))
        query_embedding = self.embedder.encode(claim, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0] # type: ignore
        top_results = torch.topk(cos_scores, k=top_k)

        # if debug:
        #     print("\n\n======================\n\n")
        #     print("Query:", query)
        #     evidence_ids = [e[1] for e in evidence]

        found = []
        data = []

        for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
                id = timeline[idx][1]

                # if debug:
                #     is_evidence = id in evidence_ids
                #     star = "(*)" if is_evidence else "\t"
                #     print(star, '\t', "(Rank: {:.0f})".format(i+1), "(Score: {:.4f})".format(score), corpus[idx])
                #     if is_evidence: found += [id]

                data.append([rumor_id, id, i+1, score.item()])

        # if debug:    
        #     for _, ev_id, ev_text in evidence:
        #         if ev_id not in found:
        #                 print('(!) ', ev_text)
        
        return data