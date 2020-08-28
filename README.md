# multi-purpose-networks
Conditioning of a neural network on multiple tasks 


# Augmentation:

No Augmentation yet for simplicity.

# Links:
- [Dataset](https://drive.google.com/file/d/1HSABKh49dAS6uCVXNp7e4iD5tkpXMH3P/view?usp=sharing)

# Experimentation Details
| Dataset       | Optimizer | Training    | Batch size | Epochs | Accuracy(%) |
|---------------|-----------|-------------|------------|--------|-------------|
| CIFAR10       | SGD       | Normal      | 128        | 350    | 95.42       |
| FASHION-MNIST | SGD       | Normal      | 128        | 350    | 94.83      |
| CIFAR10       | SGD       | Combined    | 128        | 700    | -       |
| FASHION-MNIST | SGD       | Combined    | 128        | 700    | -        |
| CIFAR10       | SGD       | Conditioned | 128        | 700    | -           |
| FASHION-MNIST | SGD       | Conditioned | 128        | 700    | -           |

# Credits

Code has been borrowed largely from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

