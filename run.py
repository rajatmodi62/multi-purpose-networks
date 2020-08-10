
import os 

os.system("python train_single_dataset.py --training_type cifar --n_epochs 350")
os.system("python train_single_dataset.py --training_type fashion_mnist --n_epochs 350")
os.system("python train_combined.py --training_type combined --n_epochs 700")
