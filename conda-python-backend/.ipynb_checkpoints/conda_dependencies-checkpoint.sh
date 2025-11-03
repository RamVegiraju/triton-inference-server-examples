conda create -y -n transformers_env python=3.10
source ~/anaconda3/etc/profile.d/conda.sh
source activate transformers_env
export PYTHONNOUSERSITE=True
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers
pip install conda-pack
conda-pack
