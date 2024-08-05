# thesis

## installing the environment

(install miniconda, checking the box too add it to PATH: https://docs.anaconda.com/free/miniconda/) and restart the command prompt

```
conda create -n clef python=3.10
conda activate clef
```

then install torch with CUDA (will probably differ on your machine; setup.py does not allow a custom index_url, so this has to be done manually):
```
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

also install custom nmslib to be able to use python 3.10 (see https://github.com/nmslib/nmslib/issues/534):
```
pip install nmslib-metabrainz
```

and then install the rest of the packages:

```
pip install -e .
```

running the ollama embeddings proxy server (in a separate terminal):
```
python -m lkae.ollama_proxy
```

## adding the clef GitLab repo to conveniently load data

at the repo root (here) execute 

```
git clone https://gitlab.com/checkthat_lab/clef2024-checkthat-lab.git
```

## Misc

note to self: had to turn off this check in pyterrier.batchretrieve _retrieve_one():
```
ln. 360        # if num_expected is not None:
ln. 361            # assert(num_expected == len(result))
```
