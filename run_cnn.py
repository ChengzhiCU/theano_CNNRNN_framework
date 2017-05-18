from network import Network
from layers import Relu, Softmax, Linear, Convolution, Pooling, Flatten
from loss import CrossEntropyLoss
from optimizer import SGDOptimizer
from solve_net import solve_net
from mnist import load_mnist_for_cnn

import theano.tensor as T

batch_size=50###batchsize=32, test will get 16 left, results in error

train_data, test_data, train_label, test_label = load_mnist_for_cnn('data')
model = Network()
model.add(Convolution('conv1', 5, 1, 8, 0.1, 28, batch_size))   # output size: N x 4 x 24 x 24
model.add(Relu('relu1'))
model.add(Pooling('pool1', 2))                  # output size: N x 4 x 12 x 12
model.add(Convolution('conv2', 5, 8, 16, 0.1, 12, batch_size))   # output size: N x 8 x 10 x 10
model.add(Relu('relu2'))
model.add(Pooling('pool2', 2))                  # output size: N x 8 x 5 x 5
model.add(Flatten('View'))                      # my own layer, to reshape the output
model.add(Linear('fc3', 16*4*4, 10, 0.1))          # input reshaped to N x 200 in Linear layer

model.add(Softmax('softmax'))

loss = CrossEntropyLoss(name='xent')

optim = SGDOptimizer(learning_rate=0.0001, weight_decay=0.005, momentum=0.9)

input_placeholder = T.ftensor4('input')
label_placeholder = T.fmatrix('label')
model.compile(input_placeholder, label_placeholder, loss, optim)

solve_net(model, train_data, train_label, test_data, test_label,
          batch_size=batch_size, max_epoch=150, disp_freq=1000, test_freq=10000)
