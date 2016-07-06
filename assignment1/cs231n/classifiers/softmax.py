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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  class_num=W.shape[1]
  train_num=X.shape[0]

  for i in range(train_num):
      f_i=X[i,:].dot(W)

      log_c=np.max(f_i)
      f_i-=log_c

      sum_i=0.0
      for f_i_j in f_i:
          sum_i+=np.exp(f_i_j)

      loss+=-f_i[y[i]]+np.log(sum_i)

      for j in range(class_num):
          p=np.exp(f_i[j])/sum_i
          dW[:,j]+=X[i,:]*(p-(j==y[i]))

  loss/=train_num
  dW/=train_num

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  class_num=W.shape[1]
  train_num=X.shape[0]

  f=np.dot(X,W)

  f-=np.max(f)

  f_correct=f[range(train_num),y]
  loss=-np.mean(np.log(np.exp(f_correct)/np.sum(np.exp(f))))

  p = np.exp(f)/np.sum(np.exp(f), axis=0)
  print 'p shape:' % p.shape
  ind = np.zeros(p.shape)
  ind[range(train_num),y] = 1
  dW = np.dot(X.T,(p-ind))
  dW /= train_num

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
