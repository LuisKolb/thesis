import re
from typing import Callable, List, NamedTuple, Tuple
from tqdm.auto import tqdm
from lkae.utils.data_loading import AuredDataset, AuthorityPost

import logging


logger = logging.getLogger(__name__)

# second logger for logging claim and evidence text score + judgement
logger_text_score = logging.getLogger('clef.verification.verify_log')

class VerificationResult(NamedTuple):
    label: str
    score: float


from abc import ABC, abstractmethod

class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, claim: str, evidence: str, **kwargs) -> VerificationResult:
        """Verify a claim based on the evidence."""
        pass
    
    def __call__(self, claim: str, evidence: str, **kwargs) -> VerificationResult:
        return self.verify(claim, evidence, **kwargs)

class Judge(object):
    scale: bool
    ignore_nei: bool
    threshold_refutes: float
    threshold_supports: float
    
    def __init__(self, scale=False, ignore_nei=True, threshold_refutes=0.15, threshold_supports=-0.15) -> None:
        self.scale = scale
        self.ignore_nei = ignore_nei
        self.threshold_refutes = threshold_refutes
        self.threshold_supports = threshold_supports

    def __call__(self, evidence_predictions: List[Tuple[str,AuthorityPost,VerificationResult]]) -> Tuple:
        return self.judge_evidence(evidence_predictions)
    
    def judge_evidence(self, evidence_predictions: List[Tuple[str,AuthorityPost,VerificationResult]]) -> Tuple:
        """
        take in claim, evidence and decision from verifier, and return the **overall** label we'll predict for the rumor

        CLEF CheckThat! task 5: score is [-1, +1] where 
        +1 means evidence strongly refutes
        -1 means evidence strongly supports
        """

        confidences = []
        predicted_evidence = []

        for claim, post, prediction in evidence_predictions:
            # if self.ignore_nei and prediction.label == "NOT ENOUGH INFO":
            #     continue

            confidence = float(prediction.score)

            # predicted confidence from verifier should always be positive
            if prediction.label == "SUPPORTS" and confidence > 0:
                # for SUPPORTS we'll flip confidence negative
                confidence *= -1
            elif prediction.label == "NOT ENOUGH INFO":
                confidence *= 0 # TODO should NEI contribute to judgement? probably not
            elif prediction.label == "REFUTES" and confidence < 0:
                # in case confidence isn't positive, flip it to positive
                confidence *= -1

            if self.scale and post.score:
                confidence = confidence * post.score # scale by retrieval score

            if not prediction.label == "NOT ENOUGH INFO":
                confidences += [confidence]
            elif self.ignore_nei:
                # prediction is NEI, skip for label calculation
                pass
            else:
                # prediction is NEI, include in label calculation
                confidences += [confidence]

            # build return list for submission
            predicted_evidence.append([
                post.url,
                post.post_id,
                post.text,
                confidence,
            ])


        if confidences:
            # mean confidence, no weighting
            meanconf = sum(confidences) / len(confidences)

            if meanconf > self.threshold_refutes:
                pred_label = "REFUTES"
            elif meanconf < self.threshold_supports:
                pred_label = "SUPPORTS"
            else:
                # evidence not conclusive enough
                pred_label = "NOT ENOUGH INFO"
        else:
            # no relevant judgements for the given evidence
            pred_label = "NOT ENOUGH INFO"

        logger.debug(f'judged {pred_label} for confidences {confidences}')

        return pred_label, predicted_evidence


def judge_using_evidence(rumor_id, claim: str, evidence: List[AuthorityPost], verifier: BaseVerifier, judge: Judge):
    evidences_with_decisions = []

    for post in evidence:
        if not post.text:
            logger.warn(f'evidence text empty for rumor with id {rumor_id}; evidence={post}')
            continue

        prediction = verifier(claim, post.text)
        evidences_with_decisions.append((claim,post,prediction))

        formatted_text = re.sub(r"\s+", " ", post.text) # replace linebreaks, etc. for pretty printing in a single line
        logger_text_score.info(f'\t{prediction} "{formatted_text}"')
        # print(f'\t{prediction} "{formatted_text}"')

    return  judge(evidences_with_decisions)


def run_verifier_on_dataset(dataset: AuredDataset, verifier: BaseVerifier, judge: Judge, blind: bool = False) -> List:
    res_jsons = []

    for i, item in enumerate(dataset):
        rumor_id = item["id"]
        if not blind: label = item["label"]
        claim = item["rumor"]

        if not item["retrieved_evidence"]:
            # only run fact check if we actually have retrieved evidence
            logger.warn(f'key "retrieved_evidence" was empty for rumor with id {claim}')
            return []
        
        retrieved_evidence = item["retrieved_evidence"] 
        
        # also log to dedicated logger for text score and judgement
        logger_text_score.info(f'({i+1}/{len(dataset)}) Verifying {rumor_id}: "{claim}"')
        # print(f'({i+1}/{len(dataset)}) Verifying {rumor_id}: "{claim}"')

        pred_label, pred_evidence = judge_using_evidence(rumor_id, claim, retrieved_evidence, verifier, judge)

        if not blind:
            logger_text_score.info(f'label:\t\t{label}')
            # print(f'label:\t\t{label}')
        
        logger_text_score.info(f'predicted:\t{pred_label}\tby {judge.__class__}')
        # print(f'predicted:\t{pred_label}\tby {judge.__class__}')
        
        logger_text_score.info('\n')
        # print('\n')

        if blind:
            res_jsons.append(
                {
                    "id": rumor_id,
                    "predicted_label": pred_label,
                    "predicted_evidence": pred_evidence,
                }
            )
        elif not blind:
            res_jsons.append(
                {
                    "id": rumor_id,
                    "label": label,
                    "claim": claim,
                    "predicted_label": pred_label,
                    "predicted_evidence": pred_evidence,
                }
            )

    from lkae.verification.models.open_ai import OpenaiVerifier
    
    if isinstance(verifier, OpenaiVerifier):
        logger_text_score.info(f'-----total token usage for verification-----')
        logger_text_score.info(f'total tokens:\t{verifier.total_tokens_used}')
        logger_text_score.info(f'prompt tokens:\t{verifier.prompt_tokens_used}')
        logger_text_score.info(f'completion tokens:\t{verifier.completion_tokens_used}')
        logger_text_score.info(f'price estimate:\t${((verifier.prompt_tokens_used/1000)*0.01) + ((verifier.completion_tokens_used/1000)*0.03)}')
        print(f'-----total token usage for verification-----')
        print(f'total tokens:\t{verifier.total_tokens_used}')
        print(f'prompt tokens:\t{verifier.prompt_tokens_used}')
        print(f'completion tokens:\t{verifier.completion_tokens_used}')
        print(f'price estimate:\t${((verifier.prompt_tokens_used/1000)*0.01) + ((verifier.completion_tokens_used/1000)*0.03)}')
    return res_jsons


def predict_evidence(dataset: AuredDataset, verifier: BaseVerifier) -> dict:
    decisions_by_id = {}
    logger_text_score.info(f'running {verifier.__class__.__name__} verifier')
    for i, item in enumerate(dataset):
        rumor_id = item["id"]
        claim = item["rumor"]

        if not item["retrieved_evidence"]:
            # only run fact check if we actually have retrieved evidence
            logger.warn(f'key "retrieved_evidence" was empty for rumor with id {claim}')
            return {}
        
        retrieved_evidence = item["retrieved_evidence"] 
        
        # also log to dedicated logger for text score and judgement
        logger_text_score.info(f'({i+1}/{len(dataset)}) Verifying {rumor_id}: "{claim}"')
        # print(f'({i+1}/{len(dataset)}) Verifying {rumor_id}: "{claim}"')

        if rumor_id not in decisions_by_id:
            decisions_by_id[rumor_id] = []

        for post in retrieved_evidence:
            
            if not post.text:
                logger.warn(f'evidence text empty for rumor with id {rumor_id}; evidence={post}')
                continue

            prediction = verifier(claim, post.text)
            decisions_by_id[rumor_id].append((post.post_id, prediction))
    
    return decisions_by_id
