# multi-purpose-networks
Conditioning of a neural network on multiple tasks 

# Links:
- [Dataset](https://drive.google.com/file/d/1HSABKh49dAS6uCVXNp7e4iD5tkpXMH3P/view?usp=sharing)
# Experimentation Details

| Dataset | Optimizer | Training | Batch size | Accuracy|
| ------- | --------- | -------- | ---------- |    |
| CIFAR10 |    SGD    |  Normal  |   128      |     | 
| FASHION-MNIST |    SGD    |  Normal  |   128      |   |
| Combined      |    SGD    |  Normal  |   128      |   |
| CIFAR10 |    SGD    |  Conditioned  |   128      |    |
| FASHION-MNIST |    SGD    |  Conditioned  |   128  |  |

# Credits

Code has been borrowed largely from [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

