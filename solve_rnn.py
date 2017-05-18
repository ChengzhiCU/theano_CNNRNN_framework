from utils import LOG_INFO
import numpy as np
import time


def one_sample_iterator(x, y, shuffle=True):
    indx = range(len(x))
    if shuffle:
        np.random.shuffle(indx)

    for i in range(0, len(x)):
        chosen_idx = indx[i]
        yield x[chosen_idx], y[chosen_idx]


def solve_rnn(model, train_x, train_y, test_x, test_y,
              max_epoch, disp_freq, test_freq):

    iter_counter = 0
    loss_list = []
    accuracy_list = []

    all_train_loss=[]
    all_train_accuracy=[]

    test_acc = []
    test_loss = []
    test_equation_acc = []

    all_test_loss=[]
    all_test_accuracy=[]
    all_test_equation_acc = []

    t = time.localtime(time.time())
    ss = time.strftime('%Y-%m-%d %H:%M:%S',t)
    print ss
    f = open(ss + 'trainloss.txt','w')
    for i in xrange(len(all_train_loss)):
        f.write("%f " % all_train_loss[i])
    f.close()
    

    for k in range(max_epoch):
        for x, y in one_sample_iterator(train_x, train_y):
            iter_counter += 1

            loss, accuracy = model.train(x, y)
            loss_list.append(loss)
            accuracy_list.append(accuracy)

            if iter_counter % disp_freq == 0:
                all_train_loss.append(np.mean(loss_list))
                all_train_accuracy.append(np.mean(accuracy_list))
                msg = 'Training iter %d, mean loss %.5f (sample loss %.5f), mean acc %.5f' % (iter_counter,
                                                                                              np.mean(loss_list),
                                                                                              loss_list[-1],
                                                                                              np.mean(accuracy_list))
                LOG_INFO(msg)
                loss_list = []
                accuracy_list = []

            if iter_counter % test_freq == 0:
                LOG_INFO('    Testing...')
                for tx, ty in one_sample_iterator(test_x, test_y, shuffle=False):
                    t_accuracy, t_equ_acc, t_loss = model.test(tx, ty)
                    test_acc.append(t_accuracy)
                    test_loss.append(t_loss)
                    test_equation_acc.append(t_equ_acc)

                msg = '    Testing iter %d, mean loss %.5f, mean acc %.5f, equation acc %.5f' % (iter_counter,
                                                                                                 np.mean(test_loss),
                                                                                                 np.mean(test_acc),
                                                                                                 np.mean(test_equation_acc))
                LOG_INFO(msg)
                all_test_loss.append(np.mean(test_loss))
                all_test_accuracy.append(np.mean(test_acc))
                all_test_equation_acc.append(np.mean(test_equation_acc))

                test_acc = []
                test_loss = []
                test_equation_acc = []
    
    #ss = 'log_LSTM_rmsprop'
    t = time.localtime(time.time())
    ss = time.strftime('%Y-%m-%d %H:%M:%S',t)
    print ss
    f = open(ss + 'trainloss.txt','w')
    for i in xrange(len(all_train_loss)):
        f.write("%f " % all_train_loss[i])
    f.close()

    f = open(ss + 'trainacc.txt','w')
    for i in xrange(len(all_train_accuracy)):
        f.write("%f " % all_train_accuracy[i])
    f.close()

    f = open(ss + 'testloss.txt','w')
    for i in xrange(len(all_test_loss)):
        f.write("%f " % all_test_loss[i])
    f.close()

    f = open(ss + 'testacc.txt','w')
    for i in xrange(len(all_test_accuracy)):
        f.write("%f " % all_test_accuracy[i])
    f.close()

    f = open(ss + 'test_equa_acc.txt','w')
    for i in xrange(len(all_test_equation_acc)):
        f.write("%f " % all_test_equation_acc[i])
    f.close()
    print 'done!!!'
