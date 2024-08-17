import os
import json
from openai import OpenAI

from lkae.verification.types import VerificationResult
from lkae.verification.verify import BaseVerifier
from lkae.verification.models._llm_sys_message import sys_message

import logging
logger = logging.getLogger(__name__)


class OpenaiVerifier(BaseVerifier):

    def __init__(self, api_key:str='', assistant_id="asst_XRITdOybDfYpIr4fVevm6qYi", temperature=0.2, top_p=1, **kwargs) -> None:
        self.client = OpenAI(
            api_key=(api_key or os.environ.get("OPENAI_API_KEY")),
        )
        self.assistant_id: str = assistant_id
        self.total_tokens_used: int = 0
        self.prompt_tokens_used: int = 0
        self.completion_tokens_used: int = 0
        
        self.system_message = sys_message
        self.temperature = temperature
        self.top_p = top_p
        
        self.valid_labels = ["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]

    
    def get_assistant_response(self, input_message):
        thread = self.client.beta.threads.create()
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=input_message,
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self.assistant_id,
            temperature=self.temperature,
            top_p=self.top_p
        )

        if run.status == 'completed':
            self.total_tokens_used += run.usage.total_tokens # type: ignore
            self.prompt_tokens_used += run.usage.prompt_tokens # type: ignore
            self.completion_tokens_used += run.usage.completion_tokens # type: ignore
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                limit=1
            )
            if messages:
                for message in messages.data:
                    if message.role == "assistant":
                        return messages.data[0].content[0].text.value # type: ignore
        else:
            logger.warn(f'run failed with status: {run.status}, returning NOT ENOUGH INFO answer')
            return '{"decision": "NOT ENOUGH INFO", confidence": 1.0}' # need to return a string that can be json-parsed

    
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        input_text = f'"{evidence}"\n\nClaim: "{claim}"'
      
        answer = self.get_assistant_response(input_text)

        if not answer:
            logger.warn(f'!!! answer was empty in response to input_text text: {input_text}')
        else:
            try:
                decision, confidence = json.loads(answer).values()
            except ValueError:
                logger.warn(f'could not json-parse response from openai model: {answer}')
                return VerificationResult("NOT ENOUGH INFO", 1.0)

        if decision and decision in self.valid_labels:
            return VerificationResult(decision, confidence)
        else:
            return VerificationResult("NOT ENOUGH INFO", 1.0)


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl
    ds = load_pkl(os.path.join(pkl_dir, 'English_train', 'pre-nam-bio.pkl'))
    sample = ds[0]

    verifier = OpenaiVerifier()

    claim = sample['rumor']
    evidence = sample['evidence'][0][2]
    print(verifier.verify(claim, evidence))