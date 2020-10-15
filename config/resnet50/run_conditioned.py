
import os 


######################## SINGLE TRAINING ###########################################################################################
#######cifar 
#os.system("python train_single_dataset.py --lr 0.1")
#os.system("python train_single_dataset.py --checkpoint_path experiments/cifar/checkpoint.pth --lr 0.01")
#os.system("python train_single_dataset.py --checkpoint_path experiments/cifar/checkpoint.pth --lr 0.001")

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

os.system("CUDA_VISIBLE_DEVICES=0 python train_conditioned.py  --backbone=resnet50 --lr 0.1")
os.system("CUDA_VISIBLE_DEVICES=0 python train_conditioned.py  --backbone=resnet50 --cifar_checkpoint_path experiments/resnet50/conditioned/cifar/checkpoint.pth --fashion_mnist_checkpoint_path experiments/resnet50/conditioned/fashion_mnist/checkpoint.pth --lr 0.01")
os.system("CUDA_VISIBLE_DEVICES=0 python train_conditioned.py  --backbone=resnet50 --cifar_checkpoint_path experiments/resnet50/conditioned/cifar/checkpoint.pth --fashion_mnist_checkpoint_path experiments/resnet50/conditioned/fashion_mnist/checkpoint.pth --lr 0.001")