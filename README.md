# CIFAR10-Image-Classification

## P1. CIFAR10 CNN: training using minibatch gradient descent algorithms

Implemented a CNN architecture with 3 convolutional layers followed by a fully connected layer of 1000 units. Each convolutional layer consists of a sublayer of 5x5 convolutional filters with stride 1 followed by a sublayer of 2x2 max-pool units with stride 2. Each neuron applies ReLU activation function.

**Task:** Evaluate and plot **the average training loss per epoch** versus the number of epoches for the training dataset, for the following optimization algorithms:
- Mini-batch gradient descent
- Mini-batch AdaGrad
- Mini-batch RMSProp
- Mini-batch gradient descent with Nesterov’s momentum
- Mini-batch Adam 
