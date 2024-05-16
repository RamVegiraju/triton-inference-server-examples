# [Adding Blog Link Here Soon]

# Setup
To execute the notebooks here will require some setup. First we create a conda environment for the conda_python backend and point towards this in the custom container we build (this removes the need for a conda pack tarball locally).

## Step 1: Conda Pack Environment Creation

```
conda create -y -n transformers_env python=3.10
source ~/anaconda3/etc/profile.d/conda.sh
source activate transformers_env
export PYTHONNOUSERSITE=True
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers
pip install conda-pack
conda-pack
```

```
bash conda_dependencies.sh
```

## Step 2: Build Docker Image
We'll refer to the image as custom-triton, this is the container we will start for the ensemble.

```
docker build -t custom-triton .
```

## Step 3: Execute onnx-exporter.ipynb notebook
Create the model.onnx file using this notebook and ensure to place it in the directory for embeddings under the sub-directory 1/. Right now we have stubbed this with a README.md.
