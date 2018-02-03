import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Multi-class Linear SVM loss function, naive implementation (with loops).
    
    In default, delta is 1 and there is no penalty term wst delta in objective function.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: a numpy array of shape (D, C) containing weights.
    - X: a numpy array of shape (N, D) containing N samples.
    - y: a numpy array of shape (N,) containing training labels; y[i] = c means
         that X[i] has label c, where 0 <= c < C.
    - reg: (float) L2 regularization strength

    Returns:
    - loss: a float scalar
    - gradient: wrt weights W, an array of same shape as W
    """
    dW = np.zeros(W.shape).astype('float') # initialize the gradient as zero
    # print (W[0])
    # print (X[:30])
    # print (y)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    # print (num_classes, num_train, sep=",")
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # print (scores)
        # print (correct_class_score)
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg*2*W

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Linear SVM loss function, vectorized implementation.
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape).astype('float') # initialize the gradient as zero
    num_train = X.shape[0]


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    scores = np.dot(X,W)
    correct_scores = scores[np.arange(scores.shape[0]),y]
    margins = np.maximum(0, scores - np.matrix(correct_scores).T + 1)
    margins[np.arange(num_train),y] = 0
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)
    # loss = np.mean(np.sum(margins,axis=1))
    # print(correct_scores)
    # print(correct_scores.shape)
    # print(type(np.matrix(correct_scores)))
    # print(type(correct_scores))
    # print(scores[0])
    # print(margins)
    # print(np.mean(np.sum(margins,axis=1)))
    # print(np.mean(np.sum(margins)))



    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    binary = margins
    binary[margins > 0] = 1
    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train),y] = -row_sum.T
    dW = np.dot(X.T, binary)
    dW /= num_train
    dW +=reg*2*W
    # dW[:,j] += X[i]
    
    # print(dW[:,0])
    # print(dW)
    # print(binary.shape)
    # print(binary)
    # print(np.sum(binary, axis=1))
    # print(binary[np.arange(num_train),y].shape)
    # print(y.shape)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
    
