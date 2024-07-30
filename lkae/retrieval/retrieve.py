from typing import Dict, List
from abc import ABC, abstractmethod
from lkae.utils.data_loading import AuredDataset

import logging
logger = logging.getLogger(__name__)

# Base class for retrieval
class EvidenceRetriever(ABC):
    def __init__(self, k):
        self.k = k
        
    @abstractmethod
    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs) -> List:
        """Retrieve documents based on the input parameters."""
        pass


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