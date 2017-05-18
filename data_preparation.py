import numpy as np


def generate_data():
    a = np.random.randint(10000)
    b = np.random.randint(10000)
    c = a + b
    while c >= 10000:
        a = np.random.randint(10000)
        b = np.random.randint(10000)
        c = a + b

    return a, b, c


def equ2feat(a, b, c):
    a_str = '%04d' % a
    b_str = '%04d' % b
    c_str = '%04d' % c

    a_feat = np.zeros((4, 10))
    b_feat = np.zeros((4, 10))
    c_feat = np.zeros((4, 10))

    for i in range(4):
        a_feat[3 - i, int(a_str[i])] = 1
        b_feat[3 - i, int(b_str[i])] = 1
        c_feat[3 - i, int(c_str[i])] = 1

    return np.hstack((a_feat, b_feat)), c_feat


def load_data(train_size=50000, test_size=10000):
    inputs = []
    labels = []
    for i in range(train_size + test_size):
        a, b, c = generate_data()
        input_feature, label_feature = equ2feat(a, b, c)
        inputs.append(input_feature)
        labels.append(label_feature)

    inputs = np.asarray(inputs)
    labels = np.asarray(labels)

    idx = range(train_size + test_size)
    np.random.shuffle(idx)

    inputs = inputs[idx]
    labels = labels[idx]

    return inputs[:train_size], labels[:train_size], inputs[train_size:], labels[train_size:]
