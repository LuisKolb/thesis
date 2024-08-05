from abc import ABC, abstractmethod
from typing import List

class EvidenceRetriever(ABC):
    def __init__(self, k):
        self.k = k
        
    @abstractmethod
    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs) -> List:
        """Retrieve documents based on the input parameters."""
        pass