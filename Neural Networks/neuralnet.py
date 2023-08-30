from readline import get_begidx
import numpy as np
import argparse
import logging, math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms


    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)



def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))

    # Implement random initialization here
    row,col=shape[0],shape[1]
    w1=np.random.uniform(-0.1,0.1,row*col).reshape((row,col))
    #w1=np.insert(w_nobias, 0, 0, axis=1)
    w1[:, 0]= 0
    return w1
    raise NotImplementedError
    

def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    row,col=shape[0],shape[1]
    w2=np.zeros((row,col))
    return w2
    raise NotImplementedError


class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """
        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        
        # self.w1 = weight_init_fn([dim1, dim2])
        self.w1 = weight_init_fn([self.n_hidden, self.n_input]) #alpha
        self.w2 = weight_init_fn([self.n_output, self.n_hidden+1]) #beta

        # initialize parameters for adagrad
        self.epsilon = 1e-5
        self.grad_sum_w1 = np.zeros([self.n_hidden, self.n_input])
        self.grad_sum_w2 = np.zeros([self.n_output, self.n_hidden+1])
#        print('here are nn shapes',self.w1.shape,self.w2.shape,self.n_input,self.n_hidden,self.n_output,self.grad_sum_w1.shape,self.grad_sum_w2.shape)
        
        self.crossEntropyTrain=np.zeros(self.n_epoch)
        self.crossEntropyValid=np.zeros(self.n_epoch)


def print_weights(nn):
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)


def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    a=np.dot(nn.w1,X.T)
#    print('a shape',a.shape,a)
    z_nobias=1/(1+np.exp(-a))
#    print('z no bias shape',z_nobias.shape,z_nobias)
    z=np.insert(z_nobias, 0, values=1, axis=0)
#    print('z shape',z.shape,z)
    b=np.dot(nn.w2,z)
#    print('b shape',b.shape,b)
    y_hat=np.exp(b)/np.sum(np.exp(b))
#    print('yhat shape',y_hat.shape,y_hat)
    return y_hat


def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
    a=np.dot(nn.w1,X.T)
    z_nobias=1/(1+np.exp(-a))
    z=np.insert(z_nobias, 0, values=1, axis=0)
    y_t=np.eye(1,10,k=y)
#    print('y_t',y_t)
    g_b=y_hat-y_t
#    print('gb shape',g_b.shape, g_b)
    d_w2,g_z=np.dot(g_b.T,z.reshape(1,z.shape[0])), np.dot(g_b,nn.w2[:,1:])
#    print('dw2 shape',d_w2.shape, d_w2,'gz shape',g_z.shape, g_z)
    g_a=np.multiply(g_z,np.multiply(z_nobias,1-z_nobias))
#    print('ga shape',g_a.shape, g_a)
    d_w1,g_x=np.dot(g_a.T,X.reshape(1,X.shape[0])), np.dot(g_a, nn.w1)
#    print('dw1 shape',d_w1.shape, d_w1, 'gx shape',g_x.shape, g_x)
    return d_w1, d_w2


def test(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    y_hat=forward(X, nn)
    label=np.argmax(y_hat, axis=0)
    predict_right=np.sum(label==y)
    error_rate=(y_hat.shape[1]-predict_right)/y_hat.shape[1]
    return label, error_rate

def crossEntropy(X, y, nn):
    crossEntropy=0
    for i in range(X.shape[0]):
            y_hat=forward(X[i,:], nn)
            y_true=np.eye(1,10,y[i])
            crossEntropy+=-np.multiply(y_true,np.log(y_hat))
    return crossEntropy

def train(X_tr, y_tr, nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    for e in range(nn.n_epoch):
        X_shuffle,y_shuffle=shuffle(X_tr, y_tr, e)
        for i in range(X_tr.shape[0]):
#            print('example',i)
            y_hat=forward(X_shuffle[i,:], nn)
            d_w1,d_w2=backward(X_shuffle[i,:], y_shuffle[i], y_hat, nn)
            nn.grad_sum_w1+=np.square(d_w1)
            nn.grad_sum_w2+=np.square(d_w2)
            nn.w1-=np.multiply(nn.lr/np.sqrt(nn.grad_sum_w1+nn.epsilon),d_w1)
            nn.w2-=np.multiply(nn.lr/np.sqrt(nn.grad_sum_w2+nn.epsilon),d_w2)
#            print(nn.w1,nn.w2)
#            y_t=np.eye(1,10,y_shuffle[i])
#            crossEntropy=-np.sum(np.multiply(y_t,np.log(y_hat)))
#            print('ent',crossEntropy)
        crossEntropyTrain=crossEntropy(X_tr, y_tr, nn)
        nn.crossEntropyTrain[e]=np.sum(crossEntropyTrain)/X_tr.shape[0]
        crossEntropyValid=crossEntropy(X_te, y_te, nn)
        nn.crossEntropyValid[e]=np.sum(crossEntropyValid)/X_te.shape[0]
#        print('epoch',e, 'crossent',nn.crossEntropyTrain, nn.crossEntropyValid)
        print('alpha',nn.w1)
        print('beta',nn.w2)
    return


if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')


    # initialize training / test data and labels
    X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics, n_epoch, n_hid, init_flag, lr=args2data(args)
#    print('here is X_tr shape',X_tr.shape, X_tr)
    # Build model
    if init_flag==1:
        weight_init_fn=random_init
    else: weight_init_fn=zero_init
    input_size=X_tr.shape[1]
    hidden_size=n_hid
    output_size=10
    nn = NN(lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size)

    # train model
    train(X_tr, y_tr, nn)
    # test model and get predicted labels and errors
    trainLabel, trainError_rate=test(X_tr, y_tr, nn)
    testLabel, testError_rate=test(X_te, y_te, nn)
    # write predicted label and error into file
    with open(out_tr, 'w') as file:
        for result in trainLabel:
            file.write(str(result) + '\n')
    with open(out_te, 'w') as file:
        for result in testLabel:
            file.write(str(result) + '\n')
    with open(out_metrics, 'w') as file:
        for i in range(nn.n_epoch):
            file.write('epoch='+str(i+1)+' crossentropy(train):'+str(nn.crossEntropyTrain[i])+'\n')
            file.write('epoch='+str(i+1)+' crossentropy(validation):'+str(nn.crossEntropyValid[i])+'\n')
        file.write('error(train): '+str(trainError_rate)+'\n')
        file.write('error(validation): '+str(testError_rate))
    
    # empirical questions
    #1a
    # y_train=[]
    # y_valid=[]
    # hiddenUnits=[5,20,50,100,200]
    # for h in hiddenUnits:
    #     nn = NN(lr, n_epoch, weight_init_fn, input_size, h, output_size)
    #     train(X_tr, y_tr, nn)
    #     y_train.append(nn.crossEntropyTrain[-1])
    #     y_valid.append(nn.crossEntropyValid[-1])
    # plt.plot(hiddenUnits, y_train, marker='o', label='Train Cross Entropy')
    # plt.plot(hiddenUnits, y_valid, marker='o', label='Validation Cross Entropy')
    # plt.title('Average Cross-Entropy vs. Hidden Units')
    # plt.xlabel('Hidden Units')
    # plt.ylabel('Average Cross-Entropy')
    # for x,y in zip(hiddenUnits, y_train):
    #     plt.text(x, y+0.03, '%.4f'%y, ha='center',va='bottom',fontsize=5)
    # for x,y in zip(hiddenUnits, y_valid):
    #     plt.text(x, y+0.03, '%.4f'%y, ha='center',va='bottom',fontsize=5)   
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    #1c
    # with open('metrics_sgd_small.txt','r') as f:
    #     data=f.readlines()
    #     list_sgd=[]
    #     for w in data:
    #         w=w.replace('\n','')
    #         list_sgd.append(w)
    # SGD=[]
    # for i in range(len(list_sgd)-2):
    #     if 'validation' in list_sgd[i]:
    #         index=list_sgd[i].rfind('.')
    #         SGD.append(float(list_sgd[i][index-1:]))

    # SGD_adgrad=nn.crossEntropyValid
    # epochs=list(range(1,101))
    # plt.plot(epochs, SGD, label='SGD')
    # plt.plot(epochs, SGD_adgrad, label='SGD with Adgrad')
    # plt.title('SGD vs. epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Validation Cross Entropy')
    # plt.grid(True)
    # plt.legend()
    # plt.show()    

    #2a
    #LR0.1 0.01 0.001
    # epochs=list(range(1,101))
    # y_train=nn.crossEntropyTrain
    # y_valid=nn.crossEntropyValid
    # plt.plot(epochs, y_train, label='Train Cross Entropy')
    # plt.plot(epochs, y_valid, label='Validation Cross Entropy')
    # plt.title('Average Cross Entropy vs. Epochs (LR=0.01)')
    # plt.xlabel('Epochs')
    # plt.ylabel('Average Cross Entropy')
    # plt.grid(True)
    # plt.legend()
    # plt.ylim((0,2.3))
    # plt.show()