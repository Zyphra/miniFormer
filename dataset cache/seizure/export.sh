#!/usr/bin/env bash
set -euo pipefail

cd /data/groups/bci/jonhuml/workspace/tokenization/miniFormer
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/miniformer-tokenization-mpl
export NUMBA_CACHE_DIR=/tmp/miniformer-seizure-numba
export PYTHONPATH=/data/groups/bci/jonhuml/workspace/neuroai/neuralset-repo:/data/groups/bci/jonhuml/workspace/neuroai/neuralfetch-repo:/data/groups/bci/jonhuml/workspace/neuroai/neuraltrain-repo:/data/groups/bci/jonhuml/workspace/neuroai/neuralbench-repo

set -o noclobber
exec /data/groups/bci/jonhuml/workspace/neuroai/neuralbench-repo/.venv/bin/python -u \
  'dataset cache/seizure/export.py' \
  --neuroai /data/groups/bci/jonhuml/workspace/neuroai \
  --data-root /data/groups/bci/datasets/small_meta_set/Dan2023Bids \
  > 'dataset cache/seizure/export.log' 2>&1
