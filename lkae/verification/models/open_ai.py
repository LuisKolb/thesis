from abc import ABC, abstractmethod
import json
from typing import List, NamedTuple
from openai import OpenAI
from openai.types.chat import ChatCompletion

import re
import os
from lkae.utils.data_loading import AuthorityPost

import logging

from lkae.verification.verify import BaseVerifier, VerificationResult
logger = logging.getLogger(__name__)


class OpenaiVerifier(BaseVerifier):
    client: OpenAI
    model: str = "gpt-4-turbo-preview"
    valid_labels: List =  [
            "REFUTES",
            "NOT ENOUGH INFO",
            "SUPPORTS"
        ]
    system_message: str = \
"""You are a helpful assistant doing simple reasoning tasks.
You will be given a statement and a claim.
You need to decide if a statement either supports the given claim ("SUPPORTS"), refutes the claim ("REFUTES"), or if the statement is not related to the claim ("NOT ENOUGH INFO"). 
USE ONLY THE STATEMENT AND THE CLAIM PROVIDED BY THE USER TO MAKE YOUR DECISION.
You must also provide a confidence score between 0 and 1, indicating how confident you are in your decision.
You must format your answer in JSON format, like this: {"decision": ["SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO"], "confidence": [0...1]}
No yapping.
""" 

    def __init__(self, api_key:str='') -> None:
        self.client = OpenAI(
            api_key=(api_key or os.environ.get("OPENAI_API_KEY")),
        )
        self.assistant_id: str = "asst_XRITdOybDfYpIr4fVevm6qYi"
        self.total_tokens_used: int = 0
        self.prompt_tokens_used: int = 0
        self.completion_tokens_used: int = 0
    
    def get_completion(self, input_message) -> ChatCompletion:
        completion = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": input_message}
            ],
            temperature=0.2
        )

        return completion
    
    def get_assistant_response(self, input_message):
        thread = self.client.beta.threads.create()
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=input_message
        )

        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=self.assistant_id,
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
            return '{"decision": "NOT ENOUGH INFO", confidence": 1.0}'

    
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