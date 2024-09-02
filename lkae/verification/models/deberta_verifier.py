import os
import torch
from transformers import pipeline, Pipeline

from lkae.verification.types import VerificationResult, BaseVerifier

import logging
logger = logging.getLogger(__name__)


class DebertaVerifier(BaseVerifier):

    def __init__(self, verifier_model: str = "tasksource/deberta-small-long-nli", **kwargs) -> None:
        # Initialize the NLI pipeline with a pre-trained model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe: Pipeline = pipeline("text-classification", model=verifier_model, device=device)

        self.label_map = {
            "CONTRADICTION": "REFUTES",
            "NEUTRAL": "NOT ENOUGH INFO",
            "ENTAILMENT": "SUPPORTS"
        }
        
    
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        # Use the NLI pipeline to predict the relationship
        result = self.pipe([dict(text=evidence, text_pair=claim)])
        pred = result[0]
        score = pred['score']
        pred_label: str = pred['label']
        parsed_label = self.label_map[pred_label.upper()]

        # Return the result
        return VerificationResult(parsed_label, score)


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl
    ds = load_pkl(os.path.join(pkl_dir, 'English_train', 'pre-nam-bio.pkl'))

    verifier = DebertaVerifier("tasksource/deberta-small-long-nli")

    for d in  ds[:5]:
        claim = d['rumor']
        evidence = d['evidence'][0][2]
        print(verifier.verify(claim, evidence))
        print('actual', d['label'])