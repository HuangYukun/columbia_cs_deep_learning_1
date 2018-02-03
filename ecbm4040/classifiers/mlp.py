from builtins import range
from builtins import object
import numpy as np

from ecbm4040.layer_funcs import *
from ecbm4040.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-2):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        hidden_layers = []
        ###################################################
        #TODO: Feedforward                                #
        ###################################################
        X = X.reshape(X.shape[0], layers[0].input_dim)
        pass_in = X
        for i in range(num_layers-1):
            hidden_layer = layers[i].feedforward(pass_in)
            pass_in = hidden_layer
            hidden_layers.append(hidden_layer)
            # print(hidden_layer.shape)
        scores = affine_forward(hidden_layer, layers[num_layers-1].params[0], layers[num_layers-1].params[1])

        if y is None:
            return scores

        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.0
        for i in range(num_layers):
            reg_loss += 0.5 * self.reg * np.sum(layers[i].params[0]**2)
        loss = data_loss + reg_loss
        
        ###################################################
        #TODO: Backpropogation                            #
        ###################################################
        dX, dW, db = affine_backward(dscores, hidden_layer, layers[num_layers-1].params[0], layers[num_layers-1].params[1])
        dW += self.reg * layers[num_layers-1].params[0]
        layers[num_layers-1].gradients = [dW, db]
        for i in range(1,num_layers):
            da = relu_backward(dX, hidden_layers[num_layers-i-1])
            # print(hidden_layers[num_layers-i-1].shape)
            # print(da.shape)
            # print(X.shape)
            # print(layers[num_layers-i-1].params[0].shape)
            # print(dX.shape)
            dX, dW, db = affine_backward(da, X, layers[num_layers-i-1].params[0], layers[num_layers-i-1].params[1])
            dW += self.reg * layers[num_layers-i-1].params[0]
            layers[i].gradients = [dW, db]
        
        ###################################################
        # TODO: Add L2 regularization                     #
        ###################################################
        for i in range(num_layers):
            square_weights += np.sum(layers.params[0]**2)
        loss += 0.5*self.reg*square_weights
        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        ###################################################
        #TODO: Use SGD or SGD with momentum to update     #
        #variables in layers                              #
        ###################################################
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        for i in range(num_layers):
            params += layers[i].params
            grads += layers[i].gradients

        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]

        for i in range(num_layers*2):
            params[i] -= learning_rate * grads[i]
        
        ###################################################
        #              END OF YOUR CODE                   #
        ###################################################
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #######################################################
        #TODO: Remember to use functions in class SoftmaxLayer#
        #######################################################
        X_n = X
        for i in range(num_layers):
            X_n = X_n.dot(layers[i].params[0])
        predictions = np.argmax(X_n,axis=1)
        
        #######################################################
        #                 END OF YOUR CODE                    #
        #######################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc
        
        


