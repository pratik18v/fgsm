# Fast Gradient Sign Method
This is the implementation (in PyTorch) of the method proposed in the paper: [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf), for generating adversarial examples.
The implementation is over the MNIST dataset.

## Results
Accuracy of the network w/o adversarial attack on the 10000 test images: 97 %

Accuracy of the network with adversarial attack on the 10000 test images: 14 %

Number of misclassified examples (as compared to clean predictions): 8374/10000
