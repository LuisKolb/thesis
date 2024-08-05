# %%
import os
import json

import pandas as pd
from IPython.display import display, HTML

from lkae.retrieval.retrieve import get_retriever, retrieve_evidence
from lkae.utils.data_loading import pkl_dir, load_pkl, root_dir

import pyterrier as pt
import pyterrier.io as ptio
import pyterrier.pipelines as ptpipelines
from ir_measures import R, MAP    

if not pt.started():
    pt.init()

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

dataset_name = 'English_train_dev_combined'
ds = datasets[dataset_name]



golden = ptio.read_qrels(os.path.join(root_dir, 'data', 'train_dev_combined_qrels.txt'))

# %%
# load each config and construct its retriever


retrievers = {}

with open('config.json', 'r') as file:
    configs = json.load(file)

    for config in configs['configs']:
        retriever_label = get_retriever(**config)
        retrievers[config['retriever_method']] = retriever_label

retrievers

# %%
# then for every variation of the dataset in ds, run the experiment with each retriever and save the results

out_dir = 'results'

data = []


for dataset_label in ds:
    for retriever_label in retrievers:
        retrieved_data = retrieve_evidence(ds[dataset_label][0:1], retrievers[retriever_label])

        pred = pd.DataFrame([[*d, retriever_label] for d in retrieved_data], columns=['qid', 'docno', 'rank', 'score', 'name']) 

        eval = ptpipelines.Evaluate(pred, golden, metrics = [R@5,MAP], perquery=False)
        r5, meanap = [v for v in eval.values()]

        score = r5

        print(f'result for retrieval run - R@5: {r5:.4f} MAP: {meanap:.4f} with config\tretriever: {retriever_label};\tds: {dataset_label}')
        
        data.append({
            'R5': r5,
            'MAP': meanap,
            'Retrieval':retriever_label, 
            'DS_Settings': dataset_label,
        })

# Convert the list of dictionaries to a DataFrame
df_retrieval = pd.DataFrame(data)

df_retrieval.to_pickle(f'{out_dir}/df_retrieval.pkl')
print('saved df!')

# Display the DataFrame
display(df_retrieval.sort_values(by='R5', ascending=False))


