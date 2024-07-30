import os
import json

from lkae.utils.data_loading import pkl_dir, load_pkl

from lkae.utils.data_loading import AuredDataset, RumorWithEvidence, eng_combined_jsonl



if __name__ == "__main__":   
    with open('config.json', 'r') as file:
        config = json.load(file) # load experiment config

    ds = load_pkl(os.path.join(pkl_dir, 'English_train', 'pre-nam-bio.pkl'))
    

    sample: RumorWithEvidence = ds[1]
    print(json.dumps(sample, indent=2))