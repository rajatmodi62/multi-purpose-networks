# multi-purpose-networks
Conditioning of a neural network on multiple tasks 


# Augmentation:

Augmentation includes random crop (after padding),horizontal flipping and tensor normalization.

# Links:
- [Dataset](https://drive.google.com/file/d/1HSABKh49dAS6uCVXNp7e4iD5tkpXMH3P/view?usp=sharing)

# Experimentation Details
| Dataset       | Optimizer | Training    | Batch size | Epochs | Accuracy(%) |
|---------------|-----------|-------------|------------|--------|-------------|
| CIFAR10       | SGD       | Normal      | 128        | 600    | 95.42       |
| FASHION-MNIST | SGD       | Normal      | 128        | 600    | 94.83      |
| CIFAR10       | SGD       | Combined    | 128        | 1200    | 87.71       |
| FASHION-MNIST | SGD       | Combined    | 128        | 1200    | 78.04        |
| CIFAR10       | SGD       | Conditioned | 128        | 1200    | **90.54**           |
| FASHION-MNIST | SGD       | Conditioned | 128        | 1200    | **94.42**           |

# Improvements 
Cifar10:2.83%
FashionMNIST:16.38%
# Credits

Code has been borrowed largely from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

