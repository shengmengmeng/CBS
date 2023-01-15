### CIFAR100N 0.2
CUDA_VISIBLE_DEVICES=0 python main_pnp.py --epoch 150 --rho-range 0.8:0.7:40 --lr 0.05 --warmup-lr 0.01 --start-expand 150 --lr-decay  cosine:20,5e-4,140

### CIFAR100N 0.8
CUDA_VISIBLE_DEVICES=0 python main_pnp.py --epoch 150 --rho-range 0.8:0.6:60 --lr 0.01 --warmup-lr 0.01 --start-expand 150 --closeset-ratio 0.8 --lr-decay  cosine:20,5e-3,140

### CIFAR80N 0.2
CUDA_VISIBLE_DEVICES=1 python main_pnp.py --epoch 150 --rho-range 0.8:0.7:40 --lr 0.05 --warmup-lr 0.001 --start-expand 130   --synthetic-data cifar80no --lr-decay  cosine:20,5e-4,140

### CIFAR80N 0.8
CUDA_VISIBLE_DEVICES=1 python main_pnp.py --epoch 150 --rho-range 0.8:0.6:60 --lr 0.01 --warmup-lr 0.01 --start-expand 100 --closeset-ratio 0.8 --synthetic-data cifar80no --lr-decay  cosine:20,5e-3,140