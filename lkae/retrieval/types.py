from abc import ABC, abstractmethod
from typing import List, NamedTuple


# list of shape [rumor_id, tweet_id, rank, similarity_score]
class EvidenceRetrieverResult(NamedTuple):
    rumor_id: str
    tweet_id: str
    rank: int
    similarity_score: float

    def __repr__(self):
        return f"EvidenceRetrieverResult(rumor_id={self.rumor_id}, tweet_id={self.tweet_id}, rank={self.rank}, similarity_score={self.similarity_score})"
    

class EvidenceRetriever(ABC):
    def __init__(self, k):
        self.k = k
        
    @abstractmethod
    def retrieve(self, rumor_id: str, claim: str, timeline: List, **kwargs) -> List[EvidenceRetrieverResult]:
        """Retrieve documents based on the input parameters."""
        pass