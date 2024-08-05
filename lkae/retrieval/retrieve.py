from typing import Dict, List
from abc import ABC, abstractmethod
from lkae.retrieval.classes import EvidenceRetriever
from lkae.utils.data_loading import AuredDataset

import logging
logger = logging.getLogger(__name__)

# Specific retriever subclasses
def retrieve_evidence(dataset: AuredDataset, retriever: EvidenceRetriever, kwargs: Dict = {}):
    data = []

    for i, item in enumerate(dataset):
        rumor_id = item["id"]
        claim = item["rumor"]
        timeline = item["timeline"]
        logger.info(f"({i+1}/{len(dataset)}) Retrieving data for rumor_id {rumor_id} using {retriever.__class__}")

        retrieved_data = retriever.retrieve(rumor_id, claim, timeline, **kwargs)
        data.extend(retrieved_data)
        logger.debug(f"retrieved data: {retrieved_data}")

    return data

def get_retriever(retriever_method: str, retriever_k: int=5, **kwargs) -> EvidenceRetriever:
    """Get the retriever object based on the method name. 
    
    Args:
        method (str): The name of the retriever method.
        k (int, optional): The number of retrieved documents. Defaults to 5.
    
    Returns:
        EvidenceRetriever: The retriever object.
    """
    if 'LUCENE' in  retriever_method.upper():
        from lkae.retrieval.methods.pyserini import LuceneRetriever
        retriever = LuceneRetriever(retriever_k, **kwargs)

    elif 'OPENAI' in retriever_method.upper():
        from lkae.retrieval.methods.open_ai import OpenAIRetriever
        retriever = OpenAIRetriever(retriever_k, **kwargs)
    
    elif 'OLLAMA' in retriever_method.upper():
        from lkae.retrieval.methods.ollama_emb import OllamaEmbeddingRetriever
        retriever = OllamaEmbeddingRetriever(retriever_k, **kwargs)

    elif 'SBERT' in retriever_method.upper():
        from lkae.retrieval.methods.sent_transformers import SBERTRetriever
        retriever = SBERTRetriever(retriever_k, **kwargs)
    
    elif 'CROSSENCODER' in retriever_method.upper():
        from lkae.retrieval.methods.sent_transformers import CrossEncoderRetriever
        retriever = CrossEncoderRetriever(retriever_k, **kwargs)

    elif 'TFIDF' in retriever_method.upper():
        from lkae.retrieval.methods.tfidf import TFIDFRetriever
        retriever = TFIDFRetriever(retriever_k, **kwargs)

    elif 'BM25' in retriever_method.upper():
        from lkae.retrieval.methods.bm25 import BM25Retriever
        retriever = BM25Retriever(retriever_k, **kwargs)

    elif 'TERRIER' in retriever_method.upper():
        from lkae.retrieval.methods.terrier import TerrierRetriever
        retriever = TerrierRetriever(retriever_k, **kwargs)

    else:
        raise ValueError(f"Invalid retriever method: {retriever_method}")

    return retriever