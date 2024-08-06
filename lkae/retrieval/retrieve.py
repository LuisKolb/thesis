from typing import Dict, List
from abc import ABC, abstractmethod
from lkae.retrieval.types import EvidenceRetriever, EvidenceRetrieverResult
from lkae.utils.data_loading import AuredDataset

import logging

logger = logging.getLogger(__name__)


def retrieve_evidence(
    dataset: AuredDataset, retriever: EvidenceRetriever, kwargs: Dict = {}
):
    data: List[EvidenceRetrieverResult] = []

    for i, item in enumerate(dataset):
        rumor_id = item["id"]
        claim = item["rumor"]
        timeline = item["timeline"]
        logger.info(
            f"({i+1}/{len(dataset)}) Retrieving data for rumor_id {rumor_id} using {retriever.__class__}"
        )

        retrieved_data = retriever.retrieve(rumor_id, claim, timeline, **kwargs)
        data.extend(retrieved_data) # extend, not append to return a flat list
        logger.debug(f"retrieved data: {retrieved_data}")

    return data


def get_retriever(
    retriever_method: str, retriever_k: int = 5, **kwargs
) -> EvidenceRetriever:
    """Get the retriever object based on the method name.

    Valid keys for the retriever_method are currently:
    BM25, TERRIER, TFIDF, LUCENE, SENT-TRANSFORMERS-HF, SENT-TRANSFORMERS-LOCAL, CROSSENCODER, OLLAMA, OPENAI

    Args:
        method (str): The name of the retriever method.
        k (int, optional): The number of retrieved documents. Defaults to 5.

    Returns:
        EvidenceRetriever: The retriever object.
    """
    if "BM25" in retriever_method.upper():
        # additional optional parameters for BM25 retriever
        # b: 0.75
        # k1: 1.6
        from lkae.retrieval.methods.bm25 import BM25Retriever
        retriever = BM25Retriever(retriever_k, **kwargs)

    elif "OLLAMA" in retriever_method.upper():
        # additional optional parameters for OllamaEmbeddingRetriever
        # model: mxbai-embed-large
        from lkae.retrieval.methods.ollama_embeddings import OllamaEmbeddingRetriever
        retriever = OllamaEmbeddingRetriever(retriever_k, **kwargs)

    elif "OPENAI" in retriever_method.upper():
        # no additional optional parameters for OpenAIRetriever
        from lkae.retrieval.methods.openai_embeddings import OpenAIRetriever
        retriever = OpenAIRetriever(retriever_k, **kwargs)

    elif "LUCENE" in retriever_method.upper():
        # additional optional parameters for LuceneRetriever
        # temp_dir_path: "./temp"
        # index_path: "./temp/index"
        # cleanup_temp_dir: True
        # nthreads: 1
        from lkae.retrieval.methods.pyserini import LuceneRetriever
        retriever = LuceneRetriever(retriever_k, **kwargs)
    
    elif "SENT-TRANSFORMERS-HF" in retriever_method.upper():
        # additional optional parameters for HFSentenceTransformersRetriever
        # model: "sentence-transformers/multi-qa-distilbert-cos-v1"
        # api_key: None
        from lkae.retrieval.methods.sent_transformers_hf import HFSentenceTransformersRetriever
        retriever = HFSentenceTransformersRetriever(retriever_k, **kwargs)

    elif "SENT-TRANSFORMERS-LOCAL" in retriever_method.upper():
        # additional optional parameters for SBERTRetriever
        # embedding_model: "all-MiniLM-L6-v2"
        from lkae.retrieval.methods.sent_transformers import SBERTRetriever
        retriever = SBERTRetriever(retriever_k, **kwargs)

    elif "CROSSENCODER" in retriever_method.upper():
        # additional optional parameters for CrossEncoderRetriever
        # model: "cross-encoder/stsb-roberta-large"
        from lkae.retrieval.methods.sent_transformers import CrossEncoderRetriever
        retriever = CrossEncoderRetriever(retriever_k, **kwargs)

    elif "TERRIER" in retriever_method.upper():
        # additional optional parameters for TerrierRetriever
        # filename: ""
        from lkae.retrieval.methods.terrier import TerrierRetriever
        retriever = TerrierRetriever(retriever_k, **kwargs)

    elif "TFIDF" in retriever_method.upper():
        # no additional optional parameters for TFIDFRetriever
        from lkae.retrieval.methods.tfidf import TFIDFRetriever
        retriever = TFIDFRetriever(retriever_k, **kwargs)

    else:
        raise ValueError(f"Invalid retriever method: {retriever_method}")

    return retriever
