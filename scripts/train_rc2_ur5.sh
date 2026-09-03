CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 --master_port=28103 train/train.py --config configs/rc2_ur5.yaml
