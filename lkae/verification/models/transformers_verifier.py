import os
import torch
from transformers import pipeline, Pipeline

from lkae.verification.types import VerificationResult, BaseVerifier

import logging
logger = logging.getLogger(__name__)


class TransformersVerifier(BaseVerifier):

    def __init__(self, verifier_model = "facebook/bart-large-mnli", task = "zero-shot-classification", **kwargs) -> None:
        # Initialize the NLI pipeline with a pre-trained model
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipe: Pipeline = pipeline(task=task, model=verifier_model, device=device)

        self.valid_labels = ["entailment", "neutral", "contradiction"]
        self.label_map = {
            "ENTAILMENT": "SUPPORTS",
            "NEUTRAL": "NOT ENOUGH INFO",
            "CONTRADICTION": "REFUTES"
        }

    def supports_token_count(self) -> bool:
        return False
        
    
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        input_text = f"{evidence}[SEP]{claim}"

        # Use the NLI pipeline to predict the relationship
        result = self.pipe(input_text, self.valid_labels)
        score = result['scores'][0]
        pred_label = result['labels'][0]
        parsed_label = self.label_map[pred_label.upper()]

        # Return the result
        return VerificationResult(parsed_label, score)


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl
    ds = load_pkl(os.path.join(pkl_dir, 'English_train', 'pre-nam-bio.pkl'))

    verifier = TransformersVerifier("FacebookAI/roberta-large-mnli", "zero-shot-classification")

    for d in  ds[:5]:
        claim = d['rumor']
        evidence = d['evidence'][0][2]
        print(verifier.verify(claim, evidence))
        print('actual', d['label'])

    print('---')

    verifier = TransformersVerifier("facebook/bart-large-mnli", "zero-shot-classification")

    for d in  ds[:5]:
        claim = d['rumor']
        evidence = d['evidence'][0][2]
        print(verifier.verify(claim, evidence))
        print('actual', d['label'])