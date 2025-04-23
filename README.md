# CIFAR10-Image-Classification

## P1. CIFAR10 CNN: training using minibatch gradient descent algorithms

Implemented a CNN architecture with 3 convolutional layers followed by a fully connected layer of 1000 units. Each convolutional layer consists of a sublayer of 5x5 convolutional filters with stride 1 followed by a sublayer of 2x2 max-pool units with stride 2. Each neuron applies ReLU activation function.

The hyper-parameter settings:
- minibatch size = 128 
- learning rate = 0.001
- total number of epoches = 100

**Evaluated and plotted the average training loss per epoch** versus the number of epoches for the training dataset, for the following optimization algorithms:
- Mini-batch gradient descent
- Mini-batch AdaGrad
- Mini-batch RMSProp
- Mini-batch gradient descent with Nesterov’s momentum
- Mini-batch Adam

## P2. CIFAR10 image classification

Designed and implemented the following convolutional neural networks for the CIFAR10 image classification task aiming to achieve a high test accuracy. Evaluated the classification accuracy by reporting top-1 and top-5 test error rates, and plotted them per epoch for the training and test dataset:

### Sequential CNN
Designed for classifying CIFAR-10 images and incorporates dropout, early stopping, L2 regularization and 5 optimizers (SGD, AdaGrad, RMSprop, Momentum, Adam) as training methods, to enhance generalization and prevent overfitting.

**Model Architecture:**
1. The architecture consists of three convolutional layers, each followed by max-pooling to progressively reduce spatial dimensions while capturing complex features. The model employs 16, 32, and 64 filters in the convolutional layers with ReLU activations for non-linearity.
2. L2 regularization is applied to convolutional and dense layers, promoting weight sparsity and reducing model complexity, considering the massive size of the training dataset.
3. The fully connected layer consists of 1000 neurons with ReLU activation, followed by a Dropout layer (rate of 0.5) to further combat overfitting. The final layer is a softmax classifier for the 10 CIFAR-10 categories.
4. Early stopping monitors validation loss, halting training when performance no longer improves, ensuring optimal generalization.

Based on these models, we proceed with the three best model architectures + training methods which give the highest accuracy on the test (val) set, or equivalently give the lowest top-1 test (val) error rate.
1. Sequential CNN with Momentum + regularization: Test Accuracy = 75.19, Test Top-1 Error Rate = 0.2481, Test Top-5 Error Rate = 0.0208
2. Sequential CNN with Momentum + regularization + early stopping: Test Accuracy = 75.16, Test Top-1 Error Rate = 0.2484, Test Top-5 Error Rate = 0.0202
3. Sequential CNN with Momentum: Test Accuracy = 69.57, Test Top-1 Error Rate = 0.3043, Test Top-5 Error Rate = 0.0338

### ResNet Models
This model architecture leverages transfer learning with the ResNet50 model, pre-trained on ImageNet, to classify images from the CIFAR-10 dataset. We further use various configurations of three training methods: dropout, regularization and 5 optimizers (SGD, AdaGrad, RMSprop, Momentum, Adam), to enhance generalization and prevent overfitting.

**Model Architecture:**
1. ResNet50's deep architecture with 50 layers makes it a strategic model choice because the layers efficiently learns complex features through residual connections, which help mitigate the vanishing gradient problem.
2. Since ResNet50 was originally trained on 224x224 pixel images, an upsampling layer is used to resize the smaller 32x32 CIFAR-10 images, ensuring compatibility with the input requirements for the ResNet model. Moreover, this upsampling significantly helped improve model performance.
3. The final classification layer from ResNet50 is removed because the ImageNet classes differ from CIFAR-10 categories. We created a custom classifier module with Global Average Pooling to reduce dimensionality while retaining spatial information, followed by Dense layers with ReLU activations for non-linearity and a softmax layer for the final 10 CIFAR-10 categories.
4. L2 regularization is applied to the dense layers and are each followed by a Dropout layer (rate = 0.5) to combat model complexity and overfitting. , promoting weight sparsity and reducing model complexity, considering the massive size of the training dataset.

Overall, using the ResNet50 functional API in Keras provided flexibility in defining more complex architectures; useful for combining transfer learning with the required customization to achieve excellent training and test results on the CIFAR-10 dataset.


# CONCLUSION: BEST MODEL FOR CIFAR-10 IMAGE CLASSIFICATION
**ResNet with AdaGrad** outperforms the Sequential CNN with Momentum + Regularization + Early Stopping by achieving higher accuracy (93.32 vs. 75.16) and lower Top-1 error rates (0.0668 vs. 0.2484), thanks to its adaptive learning rate and deeper architecture, which enhance learning efficiency and generalization. In contrast, the Sequential CNN benefits from regularization and early stopping to reduce overfitting but struggles with accuracy due to its simpler architecture.
