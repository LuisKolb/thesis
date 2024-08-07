import os
import re
import time
import requests
from collections.abc import Mapping

from transformers import AutoTokenizer

from lkae.verification.types import VerificationResult
from lkae.verification.verify import BaseVerifier
from lkae.verification.models._llm_sys_message import sys_message


class HFLlama3Verifier(BaseVerifier):
    def __init__(
        self,
        api_key: str = "",
        verifier_model: str = "meta-llama/Meta-Llama-3-70B-Instruct",
        **kwargs,
    ) -> None:
        if not verifier_model.startswith("meta-llama/"):
            # valid models:
            # meta-llama/Meta-Llama-3.1-8B-Instruct
            # meta-llama/Meta-Llama-3.1-70B-Instruct
            # meta-llama/Meta-Llama-3.1-405B-Instruct
            raise ValueError(f"Unsupported verifier model: {verifier_model}")

        self.api_key = api_key or os.environ.get("HF_API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self.API_URL = f"https://api-inference.huggingface.co/models/{verifier_model}"

        self.system_message = sys_message
        self.valid_labels = ["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            verifier_model
        )  # just needs to be any model from the llama3 family

        # models need to be warmed up (likely)
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "Wakey Wakey"},
                {"role": "user", "content": "Rise and Shine"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.query({"inputs": prompt})

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

        result = self.query({"inputs": prompt})

        if result and not isinstance(result, list) and "error" in result[0].keys():
            time.sleep(3600)  # sleep for an hour is hourly rate limit reached'
            print("hourly rate limit reached, sleeping for an hour")
            result = self.query({"inputs": prompt})

        if (
            not result
            or not len(result)
            or not isinstance(result[0], Mapping)
            or "generated_text" not in result[0]
        ):
            print(f"ERROR: unexpected answer from API: {result}")
            return VerificationResult("NOT ENOUGH INFO", float(1))

        answer = result[0]["generated_text"][len(prompt) :]

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
            print(
                f"ERROR: could not find the answer format in answer from model: {answer}"
            )
            return VerificationResult("NOT ENOUGH INFO", float(1))

    def query(self, payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        if response.status_code != 200:
            if response.status_code == 503:
                # need to warm up the model, see https://huggingface.co/docs/api-inference/quicktour#model-loading-and-latency
                res = response.json()
                print(f"Waiting for model to warm up (for {float(res['estimated_time'])} seconds)")
                time.sleep(float(res['estimated_time']))
                response = requests.post(self.API_URL, headers=self.headers, json=payload)
            else:
                raise ValueError(f"Error: {response.status_code}; Text: {response.text}")
        return response.json()


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[0]

    claim = sample["rumor"]
    evidence = sample["evidence"][0][2]

    verifier1 = HFLlama3Verifier(
        verifier_model="meta-llama/Meta-Llama-3.1-405B-Instruct"
    )
    print(verifier1.verify(claim, evidence))
    
    verifier2 = HFLlama3Verifier(
        verifier_model="meta-llama/Meta-Llama-3.1-70B-Instruct"
    )
    print(verifier2.verify(claim, evidence))

    verifier3 = HFLlama3Verifier(
        verifier_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    print(verifier3.verify(claim, evidence))


