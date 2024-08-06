import time
from exceptiongroup import catch
import requests
from collections.abc import Mapping
import os
import re

from transformers import AutoTokenizer

from lkae.verification.verify import VerificationResult, BaseVerifier


class Llama3Verifier(BaseVerifier):
    def __init__(self, api_key: str = '') -> None:
        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.headers = {"Authorization": f'Bearer {self.api_key}'}
        self.API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"

        self.system_message = """You are a helpful assistant doing simple reasoning tasks.
You will be given a statement and a claim.
You need to decide if a statement either supports the given claim ("SUPPORTS"), refutes the claim ("REFUTES"), or if the statement is not related to the claim ("NOT ENOUGH INFO"). 
USE ONLY THE STATEMENT AND THE CLAIM PROVIDED BY THE USER TO MAKE YOUR DECISION.
You must also provide a confidence score between 0 and 1, indicating how confident you are in your decision.
You must format your answer in JSON format, like this: {"decision": ["SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO"], "confidence": [0...1]}
No yapping.
"""
        self.valid_labels =  [
                "REFUTES",
                "NOT ENOUGH INFO",
                "SUPPORTS"
        ]

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") # just needs to be any model from the llama3 family

    
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        input_text = f'"{evidence}"\n\nClaim: "{claim}"'

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": input_text},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        

        result = self.query({
            "inputs": prompt
        })

        if result and not isinstance(result, list) and 'error' in result[0].keys():
            time.sleep(3600) # sleep for an hour is hourly rate limit reached'
            print('hourly rate limit reached, sleeping for an hour')
            result = self.query({
                "inputs": prompt
            })

        if not result or not len(result) or not isinstance(result[0], Mapping) or 'generated_text' not in result[0]:
            print(f'ERROR: unexpected answer from API: {result}')
            return VerificationResult("NOT ENOUGH INFO", float(1))

        answer = result[0]['generated_text'][len(prompt):]

        regex_pattern = r'{"decision": "([^"]*)", "confidence": (\d+(\.\d+)?)\}'

        match = re.search(regex_pattern, answer)

        if match:
            label = match.group(1)
            confidence = float(match.group(2))
            if label in self.valid_labels:
                return VerificationResult(label, confidence)
            else:
                print(f'ERROR: unkown label "{label}" in answer: {answer}')
                return VerificationResult("NOT ENOUGH INFO", float(1))
        else:
            print(f'ERROR: could not find the answer format in answer from model: {answer}')
            return VerificationResult("NOT ENOUGH INFO", float(1))
        

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()


#
# old code
#
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") # just needs to be any model from the llama3 family


API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"
headers = {"Authorization": f'Bearer {os.environ.get("HF_API_KEY")}'}

system_message = """You are a helpful assistant.
You need to decide if a statement by a given source either supports the given claim ("SUPPORTS"), refutes the claim ("REFUTES"), or if the premise is not related to the claim ("NOT ENOUGH INFO"). 
USE ONLY THE PROVIDED STATEMENT TO MAKE YOUR DECISION.
You must also provide a confidence score between 0 and 1, indicating how confident you are in your decision.
Format your answer in JSON format, like this: {"decision": ["SUPPORTS"|"REFUTES"|"NOT ENOUGH INFO"], "confidence": [0...1]}
Set your own temperature to 0.
No yapping.
"""

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def inference_hf_llama3(statement: str, evidence: str, model_string: str = 'instruct'):
    input_text = f'The statement: "{evidence}"\nThe claim: "{statement}"'

    valid_labels = [
        "REFUTES",
        "NOT ENOUGH INFO",
        "SUPPORTS"
    ]

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_text},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    result = query({
        "inputs": prompt
    })

    if not result or not len(result) or not isinstance(result[0], Mapping) or 'generated_text' not in result[0]:
        print(f'ERROR: unexpected answer from API: {result}')
        return ("NOT ENOUGH INFO", float(1))

    answer = result[0]['generated_text'][len(prompt):]

    regex_pattern = r'{"decision": "([^"]*)", "confidence": (\d+(\.\d+)?)\}'

    match = re.search(regex_pattern, answer)

    if match:
        label = match.group(1)
        confidence = float(match.group(2))
        if label in valid_labels:
            return (label, confidence)
        else:
            print(f'ERROR: unkown label "{label}" in answer: {answer}')
            return ("NOT ENOUGH INFO", float(1))
    else:
        print(f'ERROR: could not find the answer format in answer from model: {answer}')
        return ("NOT ENOUGH INFO", float(1))



    