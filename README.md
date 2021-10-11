# neural-network-japanese-character-recognition
## [Link to report page](https://carimo198.github.io/neural-network-japanese-character-recognition/)
This project consists of implementing various neural networks to recognise handwritten Hiragana symbols. The dataset used is Kuzushiji-MNIST (KMNIST), containing 10 Hiragana characters with 7000 samples per class.

The following networks were developed to observe the strengths and limitations of varying neural networks for image classification tasks:

- ***NetLin*** - computes a linear function of the pixels in the image, followed by log softmax. The model can be run by the command line code: python3 kuzu_main.py --net lin
- ***NetFull*** - a full connected 2-layer network with one hidden layer, plus the output layer, using tanh activation at the hidden layer and log softmax at the output layer. Run the code by typing: python3 kuzu_main.py --net full --lr=0.055 --mom=0.55
- ***NetConv*** - a convolutional neural network with two convolutional layers plus one fully connected layer, all using ReLU activation function followed by the output layer using log softmax. Max pooling and a dropout layer were also used.

Each model was run for 10 training epoch on the test set and the final accuracy and confusion matrix were saved. Please refer to the [report page](https://carimo198.github.io/neural-network-japanese-character-recognition/) for a discussion on the performance of the models and their architecture.   
