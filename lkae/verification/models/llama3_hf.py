import os
import re
import time
import json
import requests
from collections.abc import Mapping

from transformers import AutoTokenizer

from lkae.verification.models._llm_sys_message import sys_message
from lkae.verification.types import VerificationResult, BaseVerifier


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

        # retrying/sleeping is now handled in query() function automatically
        # if result and not isinstance(result, list) and "error" in result[0].keys():
        #     time.sleep(3600)  # sleep for an hour is hourly rate limit reached'
        #     print("hourly rate limit reached, sleeping for an hour")
        #     result = self.query({"inputs": prompt})

        if (
            not result
            or not len(result)
            or not isinstance(result[0], Mapping)
            or "generated_text" not in result[0]
        ):
            raise ValueError(f"ERROR: unexpected answer from API: {result}")
            return VerificationResult("NOT ENOUGH INFO", float(1))

        answer = result[0]["generated_text"][len(prompt) :]

        try:
            decision, confidence = json.loads(answer).values()
            if decision and (decision in self.valid_labels):
                return VerificationResult(decision, confidence)
            else:
                return VerificationResult("NOT ENOUGH INFO", 1.0)
        except ValueError:
            if ('\n' in answer):
                answer = answer.split('\n')[0]
                try:
                    decision, confidence = json.loads(answer).values()
                    if decision and (decision in self.valid_labels):
                        return VerificationResult(decision, confidence)
                    else:
                        return VerificationResult("NOT ENOUGH INFO", 1.0)
                except ValueError:
                    if ('\n' in answer):
                        answer = answer.split('\n')[0]
                        
                    print(f"ERROR: could not decode answer from model: {answer}")
                    return VerificationResult("NOT ENOUGH INFO", float(1))

            print(f"ERROR: could not decode answer from model: {answer}")
            return VerificationResult("NOT ENOUGH INFO", float(1))

        # regex_pattern = r'(\n|\s)*{(\n|\s)*"decision"(\n|\s)*:(\n|\s)*"([^"]*)"(\n|\s)*,(\n|\s)*"confidence"(\n|\s)*:(\n|\s)*(\d*.*.\d*)(\n|\s)*\}(\n|\s)*'

        # match = re.search(regex_pattern, answer)

        # if match:
        #     label = match.group(5)
        #     confidence = match.group(10)
        #     # remove quotes from confidence before converting, if present (for example: confidence='"1"')
        #     confidence = float(confidence.strip('"'))
        #     if label in self.valid_labels:
        #         return VerificationResult(label, confidence)
        #     else:
        #         print(f'ERROR: unkown label "{label}" in answer: {answer}')
        #         return VerificationResult("NOT ENOUGH INFO", float(1))
        # else:
        #     print(
        #         f"ERROR: could not find the answer format in answer from model: {answer}"
        #     )
        #     return VerificationResult("NOT ENOUGH INFO", float(1))


    def query(self, payload, retries=0):
        if retries > 6:
            raise ValueError(f"Error: too many retries ({retries})")
        if retries > 0:
            print(f"sleeping for {4**retries} seconds before retrying (retries={retries})")
            time.sleep(4**retries) # sleep for 4^retries seconds; up to ~ 68 minutes at 6 retries (1h is the rate limit window)
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        res_json = response.json()

        # retries if error is returned from API
        if response.status_code != 200:

            if response.status_code == 503:
                if not res_json.keys() or "estimated_time" not in res_json.keys():
                    print(f"Error (503): {response.status_code}; Text: {response.text}; retrying... (retries={retries})")
                    res_json = self.query(payload, retries+1)
                # need to warm up the model, see https://huggingface.co/docs/api-inference/quicktour#model-loading-and-latency
                print(f"Waiting for model to warm up (for {float(res_json['estimated_time'])} seconds)")
                time.sleep(float(res_json['estimated_time']))
                res_json = self.query(payload, retries=0) # reset retries to 0 since we already called sleep() in this case 

            elif response.status_code >= 400 and response.status_code < 500:
                if response.status_code == 429:
                    # rate limit reached for PRO usage, sleep for an hour
                    print(f'Error (429): {response.status_code}; Text: {response.text}; sleeping 1 hour...')
                    time.sleep(60*60*1.1)
                    res_json = self.query(payload, retries=0) # reset retries to 0 since we already called sleep() in this case 
                # some kind of 4xx error, retry after sleeping (recursively)
                print(f"Error (4xx): {response.status_code}; Text: {response.text}; retrying... (retries={retries})")
                res_json = self.query(payload, retries+1)

            else:
                raise ValueError(f"Error: {response.status_code}; Text: {response.text}")
        
        # return the response.json() dict
        return res_json


if __name__ == "__main__":
    import os
    from lkae.utils.data_loading import pkl_dir, load_pkl

    ds = load_pkl(os.path.join(pkl_dir, "English_train", "pre-nam-bio.pkl"))
    sample = ds[0]

    claim = sample["rumor"]
    evidence = sample["evidence"][0][2]

    # verifier1 = HFLlama3Verifier(
    #     verifier_model="meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
    # )
    # print(verifier1.verify(claim, evidence))

   
    verifier2 = HFLlama3Verifier(
        verifier_model="meta-llama/Meta-Llama-3.1-70B-Instruct"
    )
    print(verifier2.verify(claim, evidence))

    verifier3 = HFLlama3Verifier(
        verifier_model="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    print(verifier3.verify(claim, evidence))



