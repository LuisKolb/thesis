from typing import List
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from lkae.retrieval.classes import EvidenceRetriever

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
    

class CrossEncoderRetriever(EvidenceRetriever):
    def __init__(self, retriever_k, model="cross-encoder/stsb-roberta-large", **kwargs):
        print(f"Initializing CrossEncoderRetriever with model: {model}")
        self.model = CrossEncoder(model)
        super().__init__(retriever_k)

    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs):
    
        corpus = [t[2] for t in timeline]
        top_k = min(self.k, len(corpus))

        corpus_ranking = self.model.rank(claim, corpus, top_k, return_documents=False)

        # if debug:
        #     print("\n\n======================\n\n")
        #     print("Query:", query)
        #     evidence_ids = [e[1] for e in evidence]

        # found = []
        data = []

        for i, doc_rank_info in enumerate(corpus_ranking):
                doc = timeline[doc_rank_info['corpus_id']] # type: ignore

                # if debug:
                #     is_evidence = id in evidence_ids
                #     star = "(*)" if is_evidence else "\t"
                #     print(star, '\t', "(Rank: {:.0f})".format(i+1), "(Score: {:.4f})".format(score), corpus[idx])
                #     if is_evidence: found += [id]

                data.append([rumor_id, doc[1], i+1, doc_rank_info['score']])

        # if debug:    
        #     for _, ev_id, ev_text in evidence:
        #         if ev_id not in found:
        #                 print('(!) ', ev_text)
        
        return data
    
if __name__ == "__main__":

    retriever = CrossEncoderRetriever(5)
    data = retriever.retrieve(rumor_id='test',
                              claim='sky',
                              timeline=[['author1', 'tweet1', 'sky is blue'],
                                        ['author2', 'tweet2', 'sheesh bruh']])

    print(data)