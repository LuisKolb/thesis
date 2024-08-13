import os
import json
import jsonlines
import time
import pandas as pd
from IPython.display import display

from lkae.utils.data_loading import pkl_dir, load_pkl, root_dir, AuredDataset, AuthorityPost
from lkae.retrieval.retrieve import get_retriever,retrieve_evidence
from lkae.verification.verify import get_verifier, Judge, run_verifier_on_dataset
from lkae.utils.scoring import eval_run_custom_nofile

# import pyterrier as pt
# import pyterrier.io as ptio
# import pyterrier.pipelines as ptpipelines
# from ir_measures import R, MAP    

# if not pt.started():
#     pt.init()

datasets = {}

# walk through the pkl directory and load all the datasets in one of its subdirectories
# load each dataset with its subdirectory name and filename as the key
# skip non-pkl files
for subdir in os.listdir(pkl_dir):
    if not os.path.isdir(os.path.join(pkl_dir, subdir)):
        continue            
    datasets[subdir] = {}
    for filename in os.listdir(os.path.join(pkl_dir, subdir)):
        if not filename.endswith('.pkl'):
            continue
        key = os.path.join(subdir, filename)
        datasets[subdir][filename.split('.')[0]] = load_pkl(os.path.join(pkl_dir, key))

# possilbe splits: train, dev, train_dev_combined
# (test, all_combined don't have "labels")
split = 'dev'

dataset_split = f'English_{split}'
qrel_filename = f'{dataset_split}_qrels.txt'

dataset_variations_dict = datasets[dataset_split]
print(dataset_variations_dict.keys())

# ground truth RQ3
gold_file = os.path.join(root_dir, 'data', f'{dataset_split}.jsonl')
gold_list = [line for line in jsonlines.open(gold_file)]

# select a set of variations of the dataset
selected_variations = ["pre-nonam-bio", "nopre-nam-bio"]

# load each config and construct its retriever
setups = {}

with open('config2.json', 'r') as file:
    configs = json.load(file)

    for config in configs['configs']:
        exp_fingerprint = f'{config["retriever_method"]}__{config["verifier_method"]}'
        
        retriever = get_retriever(**config)
        verifier = get_verifier(**config)
        
        setups[exp_fingerprint] = {}
        setups[exp_fingerprint]['retriever'] = retriever
        setups[exp_fingerprint]['verifier'] = verifier

print(setups)

solomon = Judge(
    scale=False,  # ignore scaling, weigh each evidence evenly, except for confidence score given by verifier
    ignore_nei=True, # ignore NEI predictions
)

# then for every variation of the dataset in ds, run the experiment with each retriever and save the results

out_dir = 'results'
data = []


for dataset_variation in selected_variations:

    for exp_fingerprint in setups:
        # get the dataset here since it is modified in place here, contrary to RQ2
        dataset: AuredDataset = dataset_variations_dict[dataset_variation]
        start = time.time()

        retrieved_data = retrieve_evidence(dataset, setups[exp_fingerprint]['retriever'])

        dataset.add_trec_list_judgements(retrieved_data)

        verification_results = run_verifier_on_dataset(
            dataset=dataset,
            verifier=setups[exp_fingerprint]['verifier'],
            judge=solomon,
            blind=False,
        )

        # print(verification_results)

        macro_f1, strict_macro_f1 = eval_run_custom_nofile(verification_results, gold_list)

        retriever_label, verifier_label = exp_fingerprint.split('__')

        print(
            f"result for verification run - Macro-F1: {macro_f1:.4f} Strict-Macro-F1: {strict_macro_f1:.4f} with retriever: {retriever_label} and retriever: {verifier_label}"
        )

        wall_time = time.time() - start

        data.append({
            'Macro-F1': macro_f1,
            'Strict-Macro-F1': strict_macro_f1,
            'Retrieval_Method': retriever_label, 
            'Verifier_Method': verifier_label, 
            'DS_Settings': dataset_variation,
            'Time (s)': wall_time,
        })

# Convert the list of dictionaries to a DataFrame
df_verification = pd.DataFrame(data)

df_verification.to_csv(f'{out_dir}/df_verification.csv')
print(f'saved df to {out_dir}/df_verification.csv')

# Display the DataFrame
print(df_verification.sort_values(by='Macro-F1', ascending=False))