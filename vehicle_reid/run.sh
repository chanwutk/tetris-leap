# python3 tools/train_net.py --config-file configs/Taipei/bagtricks_R50-ibn.yml --num-gpus 1
export PYTHONPATH=/data/alenachao/leap/LEAP/vehicle_reid:$PYTHONPATH
CUDA_VISIBLE_DEVICES=3 python3 ./vehicle_reid/tools/plain_train_net.py --config-file configs/Base-bagtricks.yml --num-gpus 1