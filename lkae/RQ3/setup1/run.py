import os, json

from lkae.utils.data_loading import AuredDataset, RumorWithEvidence, eng_combined_jsonl

if __name__ == "__main__":   
    with open('config.json', 'r') as file:
        config = json.load(file)

    ds = AuredDataset(eng_combined_jsonl, **config)

    sample: RumorWithEvidence = ds[1]
    print(json.dumps(sample, indent=2))