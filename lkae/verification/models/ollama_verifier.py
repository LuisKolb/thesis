import json
from typing import List
from openai import OpenAI
from openai.types.chat import ChatCompletion

from lkae.verification.models._llm_sys_message import sys_message
from lkae.verification.types import VerificationResult, BaseVerifier

import logging
logger = logging.getLogger(__name__)


class OllamaVerifier(BaseVerifier):

    def __init__(self, verifier_model="llama3:instruct", temperature=0.2, top_p=1, **kwargs) -> None:
        self.client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key="ollama" # required, but unused
        )

        # self.total_tokens_used: int = 0
        # self.prompt_tokens_used: int = 0
        # self.completion_tokens_used: int = 0

        self.model = verifier_model
        self.temperature = temperature
        self.top_p = top_p

        self.system_message = sys_message
        self.valid_labels = ["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]
    

    def get_completion(self, input_message) -> ChatCompletion:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": input_message}
            ],
            temperature=self.temperature
        )

        return completion
    
    
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        input_text = f'"{evidence}"\n\nClaim: "{claim}"'

        result = self.get_completion(input_text)
        
        try:
            answer = result.choices[0].message.content
        except:
            print(f'ERROR: could not unpack response string from ollama model: {result}')
            return VerificationResult("NOT ENOUGH INFO", 1.0)
        
        try:
            if answer:
                decision, confidence = json.loads(answer).values()
            else:
                print(f'ERROR: answer was empty, parsed from result: {result}')
        except ValueError:
            print(f'ERROR: could not json-parse response from ollama model: {answer}')
            return VerificationResult("NOT ENOUGH INFO", 1.0)
        
        if decision in self.valid_labels:
            return VerificationResult(decision, confidence)
        else:
            return VerificationResult("NOT ENOUGH INFO", 1.0)
      

if __name__ == "__main__":
    verifier = OllamaVerifier()
    print(verifier.verify("I am a rumor.", "I am a statement."))