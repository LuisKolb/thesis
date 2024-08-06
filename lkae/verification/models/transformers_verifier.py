import os

from transformers import pipeline, Pipeline
import torch

from lkae.verification.verify import BaseVerifier, VerificationResult

class TransformersVerifier(BaseVerifier):
    def __init__(self, model: str = "roberta-large-mnli", **kwargs) -> None:

        # Initialize the NLI pipeline with a pre-trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe: Pipeline = pipeline("text-classification", model=model, device=device)

        self.label_map = {
            "CONTRADICTION": "REFUTES",
            "NEUTRAL": "NOT ENOUGH INFO",
            "ENTAILMENT": "SUPPORTS"
        }
    
    def verify(self, claim: str, evidence: str) -> VerificationResult:

        input_text = f"{evidence}[SEP]{claim}"

        # Use the NLI pipeline to predict the relationship
        result = self.pipe(input_text)

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
    sample = ds[0]

    verifier = TransformersVerifier()

    claim = sample['rumor']
    evidence = sample['evidence'][0][2]
    print(verifier.verify(claim, evidence))
    print(sample['label'])