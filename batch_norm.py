import numpy as np

class BatchNorm:
  """
  This is a sample code just to illustrate how batch normalization works
  i.e the forward and backward computations.
  """

  def __init__(self, x, gamma, beta, eps=1e-12):
    """
    :param x: np array, input mini-batch expected of size (N, D) where N is the
      number of inputs in the mini-batch and D is the input dimension
    :param gamma: scale parameter
    :param beta: shift parameter
    :param eps: used to avoid division by zero when variance is 0
    """

    self.x = x
    self.gamma = gamma
    self.beta = beta
    self.eps = eps
    self.cache = None

  def forward_pass(self):
    """
    Batch normalization forward pass
    :return: normalized input
    """

    N, D = self.x.shape
    mean = 1/N * np.sum(self.x, axis=0)
    x_minus_mean = (self.x - mean)
    var = 1/N * np.sum(x_minus_mean ** 2, axis=0)
    x_hat = (self.x - mean) * (var + self.eps)**(-0.5)
    out = self.gamma * x_hat + self.beta
    # cache the values to be used later in the backward pass
    self.cache = (mean, var, x_minus_mean, x_hat, out)
    return out

  def backward_pass(self, d_out):
    """
    Batch normalization backward pass
    :param d_out: output of forward pass
    :return: Gradients of x, gamma, and beta
    """

    assert self.cache is not None, 'Forward pass should run first.'
    mean, var, x_minus_mean, x_hat, out = self.cache
    N, D = d_out.shape
    x_hat_sum = np.sum(x_hat, axis=0)

    d_gamma = d_out * x_hat_sum
    d_beta = x_hat_sum

    d_x_hat = d_out * self.gamma
    i_var = (var + self.eps)**(-0.5)
    d_ivar = np.sum(d_out * x_minus_mean, axis=0)
    d_sqrtvar = d_ivar * -1/(var + self.eps)
    d_var = 0.5 * i_var * d_sqrtvar

    d1 = d_x_hat * i_var
    d2 = 2 * x_minus_mean * 1/N * np.ones((N, D)) * d_var
    d_x1 = d1+d2

    d_x2 = 1/N * np.ones((N, D)) * -1 * np.sum(d1+d2, axis=0)

    d_x = d_x1 + d_x2

    return d_x, d_gamma, d_beta