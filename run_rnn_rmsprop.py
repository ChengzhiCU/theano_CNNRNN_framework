from layers import  Softmax, Linear, RNN#, LSTM
from loss import CrossEntropyLoss
from optimizer import SGDOptimizer, RMSpropOptimizer
from network import Network
from data_preparation import load_data
from solve_rnn import solve_rnn

import theano.tensor as T

X_train, y_train, X_test, y_test = load_data()

HIDDEN_DIM = 32
INPUT_DIM = 20
OUTPUT_DIM = 10

model = Network()
model.add(RNN('rnn1', HIDDEN_DIM, INPUT_DIM, 0.1))      # output shape: 4 x HIDDEN_DIM
model.add(Linear('fc', HIDDEN_DIM, OUTPUT_DIM, 0.1))    # output shape: 4 x OUTPUT_DIM
model.add(Softmax('softmax'))

loss = CrossEntropyLoss('xent')

optim = RMSpropOptimizer(learning_rate = 0.01, rho = 0.9)
input_placeholder = T.fmatrix('input')
label_placeholder = T.fmatrix('label')

model.compile(input_placeholder, label_placeholder, loss, optim)

MAX_EPOCH = 6
DISP_FREQ = 1000
TEST_FREQ = 10000

solve_rnn(model, X_train, y_train, X_test, y_test,
          MAX_EPOCH, DISP_FREQ, TEST_FREQ)
