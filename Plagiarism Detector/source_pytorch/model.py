# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # defining 2 linear layers (fc = fully connected)
        # fc1 accepts a number of inputs 
        # and produces a number of hidden dim nodes
        self.fc1 = nn.BinaryClassifier(input_features,hidden_dim)
        
        # fc2 accepts hidden_dim as input
        # and produces a specified number of outputs
        self.fc2 = nn.BinaryClassifier(hidden_dim, output_dim)
        
        # dropout layer (probability=0.3)
        # dropout prevents overfitting of data 
        # prevents domination of a few nodes in a network
        self.drop = nn.Dropout(0.3)
        
        # adding Sigmoid layer
        # to get a class score btw 0 and 1
        self.sig = nn.Sigmoid()
        

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # Define how input(x) will move through the layers 
        # passing input(x)\banch of input features 
        # through the layers in sequence 
        
        # activation on hidden layer
        # passing through the fc1 and
        # apploying RELU activation function 
        out = F.relu(self.fc1(x))
        
        # add dropout level
        out = self.drop(out)
        
        # activate hidden layer
        out = self.fc2(out)
        
        # applying Sigmoid activation function
        # to return a final class score
        
        return self.sig(out) # returning class score 
    