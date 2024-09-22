import os
import re
import json
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential



from lkae.verification.types import VerificationResult, BaseVerifier 
from lkae.verification.models._llm_sys_message import sys_message

import logging
logger = logging.getLogger(__name__)


class Llama3AzureVerifier(BaseVerifier):

    def __init__(self, verifier_model="Meta-Llama-3.1-405B-Instruct", temperature=0.2, top_p=1, **kwargs) -> None:

        # map with valid model strings below:
        model_to_endpoint_map = {
            "Meta-Llama-3.1-405B-Instruct": "https://Meta-Llama-3-1-405B-Instruct-zpv.eastus2.models.ai.azure.com",
            "Meta-Llama-3.1-70B-Instruct": "https://Meta-Llama-3-1-70B-Instruct-hfse.eastus2.models.ai.azure.com",
            "Meta-Llama-3.1-8B-Instruct": "https://Meta-Llama-3-1-8B-Instruct-fzfwe.eastus2.models.ai.azure.com",
        }

        if verifier_model not in model_to_endpoint_map:
            raise ValueError(f"Invalid model: {verifier_model}. Valid models are: {model_to_endpoint_map.keys()}")

        self.model = verifier_model

        model_short_name = self.model.split("3.1-")[1].split("-")[0]

        self.client = ChatCompletionsClient(
            endpoint=model_to_endpoint_map[verifier_model],
            credential=AzureKeyCredential(os.environ[f'AZURE_INFERENCE_CREDENTIAL_{model_short_name}']),
        )

        self.model_to_cost_map = {
            "Meta-Llama-3.1-405B-Instruct": {"input_token_price": 0.0053, "output_token_price": 0.0016, "per_n_tokens": 1000},
            "Meta-Llama-3.1-70B-Instruct": {"input_token_price": 0.00268, "output_token_price": 0.00354, "per_n_tokens": 1000},
            "Meta-Llama-3.1-8B-Instruct": {"input_token_price": 0.0003, "output_token_price": 0.00061, "per_n_tokens": 1000},
        }

        self.total_tokens_used: int = 0
        self.prompt_tokens_used: int = 0
        self.completion_tokens_used: int = 0

        self.system_message = sys_message
        self.temperature = temperature
        self.top_p = top_p
        
        self.valid_labels = ["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]

    def supports_token_count(self) -> bool:
        return True
    
    def get_response(self, input_message):
        response = self.client.complete(
            messages=[
                SystemMessage(content=self.system_message),
                UserMessage(content=input_message),
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=1024,
        )

        try:
            self.total_tokens_used += response.usage.total_tokens
            self.prompt_tokens_used += response.usage.prompt_tokens
            self.completion_tokens_used += response.usage.completion_tokens
            return response.choices[0].message.content
        except:
            logger.warn(f'Azure API request failed for model: {self.model}, returning NOT ENOUGH INFO answer. Response: {response}')

    
    def verify(self, claim: str, evidence: str) -> VerificationResult:
        input_text = f'"{evidence}"\n\nClaim: "{claim}"'
      
        answer = self.get_response(input_text)

        if not answer:
            logger.warn(f'!!! answer was null in response to input_text text: {input_text}')
        else:
            try:
                pattern = r'{[\s\S]*}'
                # find the first match
                match = re.search(pattern, answer)
                if match:
                    decision, confidence = json.loads(match.group(0)).values()
                else:
                    raise ValueError(f'could not find json regex match in response from Azure API: {answer}')
            except ValueError:
                logger.warn(f'could not json-parse response from Azure API: {answer}, returning NOT ENOUGH INFO answer')
                return VerificationResult("NOT ENOUGH INFO", 1.0)

        if decision and (decision in self.valid_labels):
            return VerificationResult(decision, confidence)
        else:
            return VerificationResult("NOT ENOUGH INFO", 1.0)


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl
    ds = load_pkl(os.path.join(pkl_dir, 'English_train', 'pre-nam-bio.pkl'))
    sample = ds[0]

    verifier = Llama3AzureVerifier(verifier_model="Meta-Llama-3.1-405B-Instruct")

    claim = sample['rumor']
    evidence = sample['evidence'][0][2]
    print(verifier.verify(claim, evidence))

    verifier = Llama3AzureVerifier(verifier_model="Meta-Llama-3.1-70B-Instruct")

    claim = sample['rumor']
    evidence = sample['evidence'][0][2]
    print(verifier.verify(claim, evidence))

    verifier = Llama3AzureVerifier(verifier_model="Meta-Llama-3.1-8B-Instruct")

    claim = sample['rumor']
    evidence = sample['evidence'][0][2]
    print(verifier.verify(claim, evidence))