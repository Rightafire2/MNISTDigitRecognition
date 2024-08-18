# MNIST Neural Network

This repository contains code for creating a Neural Network in Python to classify handwritten digits. It is trained on the MNIST dataset from [Yann Lecun](https://yann.lecun.com/exdb/mnist/). This README walks through the mathematics behind the Neural Network and provides an explanation of the code.

## Neural Network Overview

A Neural Network is a machine learning model that mimics the human brain. Neural Networks consist of layers of neurons that connect to one another to learn patterns. These layers are connected through weights, which can be adjusted to help the network learn from its mistakes. Below, I explain the individual components and mathematics that go into making a Neural Network.

### Layers

A basic Neural Network, like the one in this repository, typically has three main types of layers: an input layer, an output layer, and hidden layers.

- **Input Layer**: This layer consists of raw input data that comes from the dataset. In the case of the MNIST dataset, it would consist of the pixel data of the handwritten image. Since the image is a 28x28 pixel image, there would be 784 datapoints and thus 784 neurons in the node.
  
- **Output Layer**: This is the final layer of the network and contains the network's predictions. In the case of MNIST classification, this would be the networks prediction of what number the handwritten image depicts. Since the dataset includes digits from 0 to 9, there are 10 neurons in the output layer.
  
- **Hidden Layers**: These layers connect the input layer to the output layer. Hidden layers can have any number of neurons, depending on the network design. They allow the network to learn more complex patterns.

### Weights and Biases

Weights and biases are instrumental in helping a network learn. They are matrices or vectors that help measure the importance of the connections between neurons in classification and are adjusted during training to minimize errors.

- **Weights**: Represent the importance of each input feature in determining the output. They are initialized randomly and updated during training.
  
- **Biases**: Allow the model to shift the activation function, providing flexibility in learning patterns.

### Activation Functions

Activation functions introduce non-linearity to the model, enabling it to learn complex patterns. Common activation functions include ReLU (Rectified Linear Unit), Leaky ReLU, Sigmoid, and Tanh. For MNIST digit classification, ReLU is often used in hidden layers, while the output layer typically uses the Softmax function. For this project, I allowed specification for ReLU, Leaky ReLU, and Softmax.

Here are the equations for the activation functions:

$$
\text{ReLU}(x) =
\begin{cases} 
x & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

The ReLU function is a linear piecewise function that captures $x$ when $x > 0$ and $0$ when $x \leq 0$. It is commonly used in image convulsion training such as this project.

$$
\text{Leaky ReLU}(x) =
\begin{cases} 
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$

The Leaky ReLU is variation of the ReLU function that opts to multiply $x$ by a really small number $\alpha$ instead of choosing $0$ when $x \leq 0$. A lot of the time, an $\alpha$ value of 0.01 is chosen in Leaky ReLU. This is the value I decided to use in my project too.

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

The Softmax function runs on an input matrix $x_i$ and computes the division of the exponential function of $x_$ and the sum of the exponential functions for all $n$ elements in the matrix.


### Forward Propagation

Forward propagation is the process of determining the output of each layer based on the input from the previous layer. Forward Propagation is vital to help the network make predictions based on the input data, weights, and biases. For this neural network, the weights and biases connect layers through the following equation:


$$
\mathbf{y} = f(\mathbf{W} \cdot \mathbf{x} + \mathbf{b})
$$

Where $\mathbf{x}$ represents the input matrix, $\mathbf{y}$ represents the output matrix, $\mathbf{W}$ represents the weight matrix, $\mathbf{b}$ represents the bias matrix. The function $f$ would be the activation function that is used on the layer. The dimensions of the matrices after the dot product would make it so that the output matrix is a vector the size of the number of neurons in the output layer. This allows us to propagate data between layers with different number of neurons.

### Backwards Propagation

Backwards propagation is the process in which the network adjusts the weights and biases based on the error from the output layer. As the name implies, backwards propagation starts from the output layer and works backwards to make the adjustments. Backward propagation computes the change in weights based on gradient of the error $\delta$ of the output layer which is known as gradient descent. In this project the initial $\delta$ is calculated through the following equation:

$$
\delta = y_{pred} - y_{true}
$$

Where $y_{pred}$ is the predicted output and $y_{true}$ is the real output that we provided to the network. From this point, the delta is calculated through the following equation:

$$
\delta^{i-1} = (f^{i})'*(W^{i})^T \cdot \delta^{i}
$$

This equation can help us iterate backwards through the layers to determine the delta that needs to be used in our gradient descent calculations. The gradient for the weights is calculated through the equation below:

$$
\nabla W^{i} = \delta^{i} \cdot (y_{pred}^{i})^T
$$

The gradient for the bias is in this case is simply the error as seen in the equation below:

$$
\nabla b^{i} = \delta^{i}
$$

Using these gradient values, we can calculate the new weights and biases in the following equations:

$$
W_i = W_i - \eta * \nabla W_{i}
$$

$$
b_i = b_i - \eta * \nabla b_{i}
$$

Where $\eta$ is the learning rate specified for the network. These are the new weights and biases used.

### Loss

Loss is a metric used in machine learning to help indicate the machine performance. There are many ways to calculate loss including MSE (Mean Squared Error), Hinge Loss, and Cross Entropy Loss. For this project, I decided to use Cross Entropy Loss which is calculated through the function below:

$$
\frac{-1}{m} * \sum_{i=0}{n} y_{true}*\log(y_{pred})
$$

Since I do have values of $0$ in $y_{pred}$ and we can not take the log of $0$, I decided to replace all $0$ values with a very small number.

## Code Walkthrough

In this section I will briefly walk through the code including how the data was manipulated into a usable form and the components of the NeuralNetwork class.

### Data Manipulation

In order to easily manipulate the data, I decided to convert the data files to CSVs. This was done through the mnist_to_csv() function here:
```
def mnist_to_csv(imgs, labels, csvPath, n, total_pixels):
    # Read image and label dat
    imgData = open(imgs, 'rb')
    labelData = open(labels, 'rb')
    # Write CSV file
    csvData = open(csvPath, 'w')

    # Skip header information and write to images array
    imgData.read(16)
    labelData.read(8)
    images = []

    # Loop through n and append pixel data to images array
    for i in range(n):
        # Capture label in image array and loop through total pixels count to add corresponding data to image
        image = [ord(labelData.read(1))]
        for j in range(total_pixels):
            image.append(ord(imgData.read(1)))
        # Append image list to images list
        images.append(image)

    # Write image data to CSV file by iterating through image array and separating row data by comma
    for image in images:
        csvData.write(','.join(str(pix) for pix in image) + '\n')

    # Close files
    imgData.close()
    labelData.close()
    csvData.close()
```

This function starts by reading the image and label data files and writes to (and creates if not already present) the CSV file. To translate the byte data to CSV, we loop through the n number of images provided in the dataset (60000 for training and 10000 for test) and the total number of pixels (784 for the 28x28 px images in the datasets) and append the appropriate label and image data as a list to the images list. By the end of the loop, we would have a list of size n with each element containing a list of 785 datapoints (1 label and 784 features - 1 for each pixel). The list is then manipulated to fit the CSV file format which separates each element of each index of the list by commas and new lines. Finally, we close the opened files. This function was derived from the [MNIST-to-CSV](https://github.com/egcode/MNIST-to-CSV) repo.

After this, we call the function on our training and test data and convert it into a pandas dataframe. We also extract the labels and columns to be used when running the Neural Network.

### NeuralNetwork Class

The NeuralNetwork class is a basic class that takes in the input size (number of neurons in the input layer), hidden layers ()


- **input_size**: The number of neurons in the input layer. In this case since we have 784 neurons, it would be 784.
  
- **hidden_layers**: The number of neurons in the hidden layers. This is taken as a list ([128, 32] would represent 128 neurons in the first hidden layer and 32 neurons in the second hidden layer)


- **output_size**: The number of neurons in the output layer. Since we have 10 neurons, this would be 10.
  
- **activation_functions**: An array of the activation functions to be used. The activation functions are static methods and should be accessed through calling the class ([NeuralNetwork.relu, NeuralNetwork.leaky_relu, NeuralNetwork.softmax] would be an example)

- **learning_rate (default=0.0001)**: The value of the learning rate to be used.

The class has methods for training the data, predicting through forward propagation, and plotting loss and accuracy metrics.

## Resources Used

- [MNIST Dataset](https://yann.lecun.com/exdb/mnist/)
- [MNIST-to-CSV](https://github.com/egcode/MNIST-to-CSV)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Backpropagation]('https://en.wikipedia.org/wiki/Backpropagation')
- [An Introduction to Gradient Descent and Backpropagation]('https://towardsdatascience.com/an-introduction-to-gradient-descent-and-backpropagation-81648bdb19b2')
- [Softmax Layer]('https://deepai.org/machine-learning-glossary-and-terms/softmax-layer')
- [A Gentle Introduction to the Rectified Linear Unit (ReLU)]('https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/')
