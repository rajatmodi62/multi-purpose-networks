
import os 


######################## SINGLE TRAINING ###########################################################################################
#######cifar 
os.system("CUDA_VISIBLE_DEVICES=1 python train_single_dataset.py --backbone=resnest50 --lr 0.1")
os.system("CUDA_VISIBLE_DEVICES=1 python train_single_dataset.py --backbone=resnest50 --checkpoint_path experiments/resnest50/cifar/checkpoint.pth --lr 0.01")
os.system("CUDA_VISIBLE_DEVICES=1 python train_single_dataset.py --backbone=resnest50 --checkpoint_path experiments/resnest50/cifar/checkpoint.pth --lr 0.001")

###fashion-mnist 
# os.system("python train_single_dataset.py --training_type fashion_mnist --lr 0.1")
# os.system("python train_single_dataset.py --training_type fashion_mnist --checkpoint_path experiments/fashion_mnist/checkpoint.pth --lr 0.01")
# os.system("python train_single_dataset.py --training_type fashion_mnist --checkpoint_path experiments/fashion_mnist/checkpoint.pth --lr 0.001")



######################## COMBINED TRAINING ###########################################################################################
##combined training 

# os.system("python train_combined.py  --lr 0.1")
# os.system("python train_combined.py  --checkpoint_path experiments/combined/cifar/checkpoint.pth --lr 0.01")
# os.system("python train_combined.py  --checkpoint_path experiments/combined/cifar/checkpoint.pth --lr 0.001")


######################## CONDITIONED TRAINING ###########################################################################################

# os.system("python train_conditioned.py  --lr 0.1")
# os.system("python train_conditioned.py  --cifar_checkpoint_path experiments/conditioned/cifar/checkpoint.pth --fashion_mnist_checkpoint_path experiments/conditioned/fashion_mnist/checkpoint.pth --lr 0.01")
# os.system("python train_conditioned.py  --cifar_checkpoint_path experiments/conditioned/cifar/checkpoint.pth --fashion_mnist_checkpoint_path experiments/conditioned/fashion_mnist/checkpoint.pth --lr 0.001")