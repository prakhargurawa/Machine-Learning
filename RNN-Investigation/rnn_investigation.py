from tensorflow import keras
import keras.backend as K
import numpy as np

# Theoritical tutorial : https://www.simplilearn.com/tutorials/deep-learning-tutorial/rnn

# How do basic RNNs work (in Keras)?
#
# A SimpleRNN layer in Keras, with n units (ie output of size n), has
# a state vector h_t of size h_dim. It gets updated at each
# time-step. It provides this as an output also, so in fact h_dim =
# n.
#
# At every time-step, the input x[t] comes in and combines with the
# previous state to give the new state, as follows:
#
# h_t = np.tanh(x[t] @ W_xh + h_t @ W_hh + b)
#
# A SimpleRNNCell is basically this calculation.
#
# A SimpleRNN layer runs a SimpleRNNCell once per time-step, storing
# the state.
#
# To demonstrate exactly what happens inside an RNN and an RNN "cell",
# we're going to get the same results three ways:
#
# 1. Run matrix multiplication and tanh and iterate by hand
# 2. SimpleRNNCell iterated by hand
# 3. SimpleRNN
#


def RNN_manual(y_dim, W_xh, W_hh, b, x, h0):
    
    # in this function, the input x is one sequence. no batches.

    T = x.shape[0] # number of time-steps
    x_dim = x.shape[1] # dimensionality of input
    h_dim = h0.shape[0] # dimensionality of state
    assert W_xh.shape[0] == x.shape[1]
    # dimensionality of state = dimensionality of output
    assert W_xh.shape[1] == W_hh.shape[0] == W_hh.shape[1] == b.shape[0] == h_dim == y_dim 

    y = np.zeros((T, y_dim))
    h_t = h0
    for t in range(T):
        y[t] = h_t = np.tanh(x[t] @ W_xh + h_t @ W_hh + b)
    return y


def RNNCell_manual(y_dim, W_xh, W_hh, b, x, h0):
    
    # in this function, we process a batch. so x and h0 have an
    # initial batch dim

    T = x.shape[1] # number of time-steps
    x_dim = x.shape[2] # dimensionality of input
    h_dim = h0.shape[1] # dimensionality of state
    assert W_xh.shape[0] == x.shape[2]
    # dimensionality of state = dimensionality of output
    assert W_xh.shape[1] == W_hh.shape[0] == W_hh.shape[1] == b.shape[0] == h_dim == y_dim 

    rnnc = keras.layers.SimpleRNNCell(y_dim, input_shape=(x_dim,))

    # run once so it can understand what weight shapes it should have,
    # so we can run set_weights, but throw away result
    rnnc(x[:, 0, :], h0) 
    rnnc.set_weights((W_xh, W_hh, b))
        
    y = np.zeros((T, y_dim))
    h_t = h0
    for t in range(T):
        y[t], h_t = rnnc(x[:, t, :], h_t)
    return y


def RNN(y_dim, W_xh, W_hh, b, x, h0):

    # again, we process a batch

    # return_sequences=True, so we can see every output
    rnn = keras.layers.SimpleRNN(y_dim, return_sequences=True)
    # run once so it can understand what weight shapes it should have,
    # so we can run set_weights, but throw away result
    rnn(x[:, 0:1, :])
    rnn.set_weights((W_xh, W_hh, b))
    rnn.states = h0
    y = rnn(x)
    return y
    


def see_RNN_weights(y_dim, x):

    # this is just to prove to ourselves what shapes the
    # vectors/matrices have

    # return_sequences=True, so we can see every output. also
    # return_state=True, so we get both state and output (which are
    # the same in the SimpleRNN, but different in other models!)
    rnn = keras.layers.SimpleRNN(y_dim, return_sequences=True, return_state=True)
    # run the rnn: pass input sequence, get back output sequence and state sequence
    y, h = rnn(x)
    W_xh, W_hh, b = rnn.get_weights()
    # print weights to confirm the shapes
    # these will be randomly initialised, but interesting pattern
    print("x", x.shape)
    print("h", h.shape)
    print("W_xh", W_xh.shape)
    print("W_hh", W_hh.shape)
    print("b", b.shape)
    print("y", y.shape)


# There are a lot of different vectors/matrices of different
# sizes/shapes: inputs, various weights and biases, state,
# outputs. But in the end, there are only three distinct size values:
#
# x_dim (size of input vector)
# T (number of time-steps)
# y_dim (size of output vector, equal to size of hidden state)
# batch_size
#
# As we can see:
# x has shape (batch_size, T, x_dim)
# h has shape (batch_size, y_dim)
# W_xh has shape (x_dim, y_dim)
# W_hh has shape (y_dim, y_dim)
# b has shape (y_dim,)
# y has shape (batch_size, T, y_dim) (assuming we ask for output at every time-step)
    


x_dim = 3
y_dim = 2 # "units"
batch_size = 1
h0 = np.array([0, 0.])
W_xh = np.array([[0, 0],
                 [0.5, 0.5],
                 [1, 1]])
W_hh = np.array([[1, 0.1],
                 [0, 1.]])
b = np.array([0, 0.4])
x = np.array([[0, 0, 0],
              [1, 1, 1],
              [0, 0, 0],
              [-1, -1, -1],
              [0, 0, 0.]]) # 5 time-steps, x_dim=3

# 1. manual: we'll use plain Numpy, no batch dimension
print(RNN_manual(y_dim, W_xh, W_hh, b, x, h0))

# for 2-3, change to tensor and add batch dimension (=1)
h0 = K.constant(h0.reshape((batch_size, *h0.shape)))
x = K.constant(x.reshape((batch_size, *x.shape)))

# 2. manual iteration of a "cell"
print(RNNCell_manual(y_dim, W_xh, W_hh, b, x, h0))

# 3. layer
print(RNN(y_dim, W_xh, W_hh, b, x, h0))
      
# confirm: what weight shapes?
see_RNN_weights(y_dim, x)

