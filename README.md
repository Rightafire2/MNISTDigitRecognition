
# MNIST Neural Network

This repo contains code for creating a Neural Network in Python to classify handwritten digits. It is trained on the MNIST dataset from [Yann Lecun](https://yann.lecun.com/exdb/mnist/). I will first walk through the mathematics behind the Neural Network and follow with an explanation of the code.

## Neural Network Overview

A Neural Network is a machine learning (ML) model that works similar to how the human brain works. Neural Networks contain layers of neurons which connect to one another in order to learn. These layers are connected through weights which can be manipulated to help the network learn from its mistakes. In this section I will briefly walk through the indivudal components and mathematics that go into making a Neural Network.

### Layers

A very basic Neural Network, like the network in this repo, could have three main types of layers: an input layer, an output layer, and hidden layers. The input layer would consist of the raw input data that comes from the dataset. In the case of the MNIST dataset, it would consist of the pixel data of the handwritten image. Since the image is a 28x28 pixel image, there would be 784 datapoints and thus 784 neurons in the node. The output layer is the last layer in the network. It consists of the network output. In the case of MNIST classification, this would be the networks prediction of what number the handwritten image depicts. Since the dataset covers number 0 through 9, this layer would have one neuron for each possible prediction, thus 10 neurons. The hidden layer is the layer that connects the input layer to the output layer. These layers can have any number of neurons specified by the creator of the network. The hidden layers allow for abstraction so the network can have a more intricate method of learning.

### Weights and Biases

Weights and biases are instrumental in helping a network learn. They are a key component in connecting layers. Weights are numerical representations of how important each neuron connection. Biases are extra parameters that are help the network learn. They can be used to avoid 0 outputs, horizontally shift activation functions to avoid a sole reliance on weights for better fitting, allow for more complex learning patters, and more. For this neural network, the weights and biases connect layers through the following equation:


$$
\mathbf{y} = f(\mathbf{W} \cdot \mathbf{x} + \mathbf{b})
$$
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\bg{white}\mathbf{y}=f(\mathbf{W}\cdot\mathbf{x}&plus;\mathbf{b})" title="\mathbf{y}=f(\mathbf{W}\cdot\mathbf{x}+\mathbf{b})" alt="Layer-Wise Equation" />>
</p>

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

