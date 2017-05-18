from network import Network
from layers import Relu, Sigmoid, Softmax, Linear, Convolution
from loss import CrossEntropyLoss
from optimizer import SGDOptimizer
from solve_net import solve_net
from mnist import load_mnist_for_mlp,load_mnist_for_cnn

import theano.tensor as T

train_data, test_data, train_label, test_label = load_mnist_for_mlp('data')# load_load_mnist_for_cnn('data')mnist_for_mlp('data')
model = Network()
#model.add(Convolution('conv1', 5, 1, 2, 0.1, 28, 32))   # output size: N x 4 x 24 x 24
#model.add(Relu('relu1'))
model.add(Linear('fc1', 784, 256, 0.01))
model.add(Relu('relu1'))
model.add(Linear('fc2', 256, 10, 0.01))
model.add(Softmax('softmax'))

loss = CrossEntropyLoss(name='xent')

optim = SGDOptimizer(learning_rate=0.001, weight_decay=0.005, momentum=0.9)

input_placeholder = T.fmatrix('input')
label_placeholder = T.fmatrix('label')
model.compile(input_placeholder, label_placeholder, loss, optim)

solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=32, max_epoch=100, disp_freq=1000, test_freq=10000)
