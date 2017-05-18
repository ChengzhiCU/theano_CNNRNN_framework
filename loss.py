import theano.tensor as T
import theano

class CrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, inputs, labels):
        # Your codes here
        # hint: labels are already in one-hot form
        #return -T.mean(T.log(inputs)[T.arange(labels.shape[0]),labels])
        #return -T.mean(T.log(inputs) * labels)
        return T.mean(theano.tensor.nnet.nnet.categorical_crossentropy(inputs,labels))