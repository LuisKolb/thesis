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