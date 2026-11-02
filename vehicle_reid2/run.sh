#!/usr/bin/env bash
# Usage: bash ./vehicle_reid2/run.sh <dataset_name> [config_file] [gpu_id] [output_dir] [dataset_root]
#   dataset_name : e.g. caldot1, caldot2 (required)
#   config_file  : path to .yml config (default: vehicle_reid2/configs/<Dataset>/bagtricks_R50-ibn.yml)
#   gpu_id       : CUDA device id (default: 0)
#   output_dir   : writable directory for logs/checkpoints (default: /data/alenachao/data/leap-dataset/logs/<dataset_name>)
#   dataset_root : root directory containing your dataset folders (default: datasets)
#
# Example:
#   bash ./vehicle_reid2/run.sh caldot1
#   bash ./vehicle_reid2/run.sh caldot1 vehicle_reid2/configs/Caldot1/bagtricks_R50-ibn.yml 1
#   bash ./vehicle_reid2/run.sh caldot1 "" 0 /data/alenachao/data/leap-dataset/logs/caldot1 /data/alenachao/data/leap-dataset/datasets
#   bash ./vehicle_reid2/run.sh amsterdam "" 6 /data/alenachao/data/leap-dataset/logs/amsterdam /data/alenachao/data/leap-dataset/datasets

DATASET=${1:?"Error: dataset_name is required. Usage: bash run.sh <dataset_name> [config_file] [gpu_id] [output_dir] [dataset_root]"}
CONFIG=${2:-"vehicle_reid2/configs/$(python3 -c "print('${DATASET}'.capitalize())")/bagtricks_R50-ibn.yml"}
GPU=${3:-0}
OUTPUT_DIR=${4:-"/data/alenachao/data/leap-dataset/logs/${DATASET}"}
DATASET_ROOT=${5:-"datasets"}

export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)":$PYTHONPATH
export FASTREID_DATASETS="$DATASET_ROOT"

mkdir -p "$OUTPUT_DIR"

echo "Dataset      : $DATASET"
echo "Config       : $CONFIG"
echo "GPU          : $GPU"
echo "Output dir   : $OUTPUT_DIR"
echo "Dataset root : $DATASET_ROOT"

CUDA_VISIBLE_DEVICES=$GPU python3 vehicle_reid2/tools/train_net.py \
  --config-file "$CONFIG" \
  --num-gpus 1 \
  OUTPUT_DIR "$OUTPUT_DIR"