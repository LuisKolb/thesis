import os
import pickle as pkl

from lkae.utils.data_loading import AuredDataset, root_dir

def save_pkl(dir_path, file_name, ds: AuredDataset):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fp = os.path.join(dir_path, f'{file_name}.pkl')
    with open(fp, 'wb') as file:
        pkl.dump(ds, file)
        print(f'saved dataset to {fp}')

if __name__ == "__main__":
    file_names = ['English_train.jsonl', 'English_dev.jsonl', 'English_train_dev_combined.jsonl', 'English_all_combined.jsonl']
    for file_name in file_names:
        for pre in [True, False]:
            for nam in [True, False]:
                for bio in [True, False]:
                    config = {
                        'preprocess': pre,
                        'add_author_name': nam,
                        'add_author_bio': bio,
                        'author_info_filepath': os.path.join(root_dir, 'data', 'combined-author-data-translated.json'),
                    }

                    fingerprint = 'pre-' if config['preprocess'] else 'nopre-'
                    fingerprint += 'nam-' if config['add_author_name'] else 'nonam-'
                    fingerprint += 'bio' if config['add_author_bio'] else 'nobio'

                    ds = AuredDataset(os.path.join(root_dir, 'data', file_name), **config)

                    save_pkl(os.path.join(root_dir, 'index', file_name.split('.')[0]),
                            fingerprint,
                            ds)