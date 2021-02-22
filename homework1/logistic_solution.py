import json
import numpy


'''
Do not change anything in the input and output format. 
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is logistic_train(), logistic_test() and feature_sqaure().
'''

def sigmoid(x):

    z = 1.0 / (1.0 + numpy.exp((-1) * x))
    return z

# Q5.1
def logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations):
    

    Xtrain = numpy.array(Xtrain)
    ytrain = numpy.array(ytrain)
    m, n = Xtrain.shape
    for i in range(0, max_iterations):
        # print(i)
        pred = sigmoid(numpy.dot(Xtrain, w) + b)
        b += (-1) * (1.0 / m) * sum(pred - ytrain) * step_size
        w += numpy.array([(-1) * (1.0 / m) * step_size * (sum((pred - ytrain) * Xtrain[:, j])) for j in range(n)])

    return w, b

# Q5.2

def logistic_test(Xtest, ytest, w, b):

    X = numpy.array(Xtest)

    pred = sigmoid(numpy.dot(X, w) + b)
    numpy.putmask(pred, pred >= 0.5, 1.0)
    numpy.putmask(pred, pred < 0.5, 0.0)

    test_acc = float(sum(pred == ytest)) / len(ytest)

    return test_acc


# Q5.3
def feature_square(Xtrain, Xtest):
    '''
    - Xtrain: training features, consists of num_train data points, each of which contains a D-dimensional feature
    - Xtest: testing feature, consists of num_test data, each of which contains a D-dimensional feature

    Returns:
    - element-wise squared Xtrain and Xtest.
    '''
    Xtrain_s = numpy.square(Xtrain)
    Xtest_s = numpy.square(Xtest)
    return Xtrain_s, Xtest_s

#============================END=====================================

'''
Please DO NOT CHANGE ANY CODE below this line.
You should only write your code in the above functions.
'''

def data_loader_toydata(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, test_set = data_set['train'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    return Xtrain, ytrain, Xtest, ytest

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = 0
        else:
            ytrain[i] = 1
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = 0
        else:
            ytest[i] = 1

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest

def inti_parameter(Xtrain):
    m, n = numpy.array(Xtrain).shape
    w = numpy.array([0.0] * n)
    b = 0
    step_size = 0.1
    max_iterations = 500
    return w, b, step_size, max_iterations

def logistic_toydata1():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata1.json')

    w, b, step_size, max_iterations = inti_parameter(Xtrain)

    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)
    
    return test_acc

def logistic_toydata2():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata2.json')

    w, b, step_size, max_iterations = inti_parameter(Xtrain)

    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)
    
    return test_acc

def logistic_toydata2s():

    Xtrain, ytrain, Xtest, ytest = data_loader_toydata(dataset = 'toydata2.json') # squared data

    Xtrain_s, Xtest_s = feature_square(Xtrain, Xtest)
    w, b, step_size, max_iterations = inti_parameter(Xtrain_s)


    w_l, b_l = logistic_train(Xtrain_s, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest_s, ytest, w_l, b_l)

    return test_acc

def logistic_mnist():

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')
    w, b, step_size, max_iterations = inti_parameter(Xtrain)

    w_l, b_l = logistic_train(Xtrain, ytrain, w, b, step_size, max_iterations)
    test_acc = logistic_test(Xtest, ytest, w_l, b_l)

    return test_acc

def main():

    test_acc = dict()

    #=========================toydata1===========================
    
    test_acc['toydata1'] = logistic_toydata1() # results on toydata1
    print('toydata1, test acc = %.4f \n' % (test_acc['toydata1']))

    #=========================toydata2===========================

    test_acc['toydata2'] = logistic_toydata2() # results on toydata2

    print('toydata2, test acc = %.4f \n' % (test_acc['toydata2']))
    test_acc['toydata2s'] = logistic_toydata2s() # results on toydata2 but with feature squared

    print('toydata2 w/ squared feature, test acc = %.4f \n' % (test_acc['toydata2s']))
    #=========================mnist_subset=======================

    test_acc['mnist'] = logistic_mnist() # results on mnist
    print('mnist test acc = %.4f \n' % (test_acc['mnist']))

     
    with open('logistic.json', 'w') as f_json:
        json.dump([test_acc], f_json)

if __name__ == "__main__":
    main()
