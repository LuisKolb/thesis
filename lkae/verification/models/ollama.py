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



class OllamaVerifier(BaseVerifier):
    client: OpenAI
    model: str = "gpt-4-turbo-preview"
    valid_labels: List =  [
            "REFUTES",
            "NOT ENOUGH INFO",
            "SUPPORTS"
        ]
    system_message: str = """You are a helpful assistant doing simple reasoning tasks.
You will be given a statement and a claim.
You need to decide if a statement either supports the given claim ("SUPPORTS"), refutes the claim ("REFUTES"), or if the statement is not related to the claim ("NOT ENOUGH INFO"). 
USE ONLY THE STATEMENT AND THE CLAIM PROVIDED BY THE USER TO MAKE YOUR DECISION.
You must also provide a confidence score between 0 and 1, indicating how confident you are in your decision.
You must format your answer in JSON format, like this: {"decision": ["SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO"], "confidence": [0...1]}
No yapping.
""" 

    def __init__(self, api_key:str='') -> None:
        self.client = OpenAI(
            base_url = 'http://localhost:11434/v1',
            api_key="ollama" # required, but unused
        )
        self.total_tokens_used: int = 0
        self.prompt_tokens_used: int = 0
        self.completion_tokens_used: int = 0
    
    def get_completion(self, input_message) -> ChatCompletion:
        completion = self.client.chat.completions.create(
            model="llama3:instruct",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": input_message}
            ],
            temperature=0.2
        )

        return completion
    
    
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        input_text = f'"{evidence}"\n\nClaim: "{claim}"'

        valid_labels = [
            "REFUTES",
            "NOT ENOUGH INFO",
            "SUPPORTS"
        ]

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
        if decision in valid_labels:
            return VerificationResult(decision, confidence)
        else:
            return VerificationResult("NOT ENOUGH INFO", 1.0)
      
if __name__ == "__main__":
    verifier = OllamaVerifier()
    print(verifier.verify("I am a rumor.", "I am a statement."))