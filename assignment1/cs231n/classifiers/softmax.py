import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
    """
  # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = np.exp(X.dot(W))
   
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    correct_scores = np.array([scores[i, y[i]] for i in range(num_train)])
    
    sum_scores = np.sum(scores, axis = 1)
    prob = np.log(np.divide(correct_scores, sum_scores))
    loss = sum(correct_scores)    
    reg_loss = (reg * np.sum(W * W))
    loss /= num_train
    loss += reg_loss
    
    for i in range(num_train):
        dW[:, np.argmax(scores[i])] += X[i]
        
    dW /= num_train
    dW += (reg * W)
            
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
    """
  # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = np.exp(X.dot(W))
    print(X.shape)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    correct_scores = np.array([scores[i, y[i]] for i in range(num_train)])
    
    sum_scores = np.sum(scores, axis = 1)
    prob = np.log(np.divide(correct_scores, sum_scores))
    loss = sum(correct_scores)    
    reg_loss = (reg * np.sum(W * W))
    loss /= num_train
    loss += reg_loss
    
    for i in range(num_train): 
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += ((scores[i, j]/sum_scores[i]) - 1)*X[i] + (reg * W[:,j])
            else: 
                dW[:, j] += ((scores[i, j]/sum_scores[i]))*X[i] + (reg * W[:,j])
        
    dW /= num_train
    dW += (reg * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW

