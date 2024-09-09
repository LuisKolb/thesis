from abc import ABC, abstractmethod
from typing import NamedTuple


class VerificationResult(NamedTuple):
    label: str
    score: float


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, claim: str, evidence: str, **kwargs) -> VerificationResult:
        """Verify a claim based on the evidence."""
        pass

    def __call__(self, claim: str, evidence: str, **kwargs) -> VerificationResult:
        return self.verify(claim, evidence, **kwargs)
    
    @abstractmethod
    def supports_token_count(self) -> bool:
        pass

    total_tokens_used: int = 0
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0
    model_to_cost_map: dict = {}
    model: str = ""