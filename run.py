
import os 

# os.system("python train_single_dataset.py --training_type cifar --n_epochs 350")
# os.system("python train_single_dataset.py --training_type fashion_mnist --n_epochs 350")
# os.system("python train_combined.py --training_type combined --n_epochs 700")


#######cifar 
# os.system("python train_single_dataset.py --lr 0.1")
# os.system("python train_single_dataset.py --checkpoint_path experiments/cifar/checkpoint.pth --lr 0.01")
# os.system("python train_single_dataset.py --checkpoint_path experiments/cifar/checkpoint.pth --lr 0.001")

###fashion-mnist 
os.system("python train_single_dataset.py --training_type fashion_mnist --lr 0.1")
os.system("python train_single_dataset.py --training_type fashion_mnist --checkpoint_path experiments/fashion_mnist/checkpoint.pth --lr 0.01")
os.system("python train_single_dataset.py --training_type fashion_mnist --checkpoint_path experiments/fashion_mnist/checkpoint.pth --lr 0.001")