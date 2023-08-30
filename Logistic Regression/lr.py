import numpy as np
import sys
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

# get the label and feature array from the formatted tsv file:
def getLabelAndFeature(file):
    data=np.loadtxt(file,delimiter='\t', comments=None, encoding='utf-8')
    y=data[:,0]
    interceptFeature=np.ones((y.shape[0],1))
    X=np.hstack((interceptFeature,data[:,1:]))
    return X,y

def ge(file):
    data=np.loadtxt(file,delimiter=' ', comments=None, encoding='utf-8')
    y=data[:,0]
    interceptFeature=np.ones((y.shape[0],1))
    X=np.hstack((interceptFeature,data[:,1:]))
    return X,y

# Train the logistic regression model using SGD:
def train(theta, X, y, num_epoch, learning_rate, X1, y1):
    #parameterTheta=np.zeros((X.shape[1],))
    N=y.shape[0]
    N1=y1.shape[0]
    negLogLikelihood=[]
    vNeglogLikelihood=[]
    for epoch in range(num_epoch):
        likelihood=0
        vLikelihood=0
        for i in range(N):
            #SGD=(-X[i,:]*(y[i]-sigmoid(theta@X[i,:])))/N
            #for j in range(X.shape[1]):
            #    SGD=-X[i,j]*(y[i]-sigmoid(theta@X[i,:]))/N
            #    theta[j]=theta[j]-learning_rate*SGD
            #theta-=learning_rate*SGD
            theta+=learning_rate*X[i,:]*(y[i]-sigmoid(theta@X[i,:]))
            likelihood+=-y[i]*(theta@X[i,:])+np.log(1+np.exp(theta@X[i,:]))
        for j in range(N1):
            vLikelihood+=-y1[j]*(theta@X1[j,:])+np.log(1+np.exp(theta@X1[j,:]))
        negLogLikelihood.append(likelihood/N)
        vNeglogLikelihood.append(vLikelihood/N1)
    return theta, negLogLikelihood, vNeglogLikelihood

# Predict the value:
def predict(theta, X):
    predictLabel=[]
    for i in range(X.shape[0]):
        probability=sigmoid(theta@X[i,:])
        if probability >=0.5:
            predictLabel.append(1)
        else: 
            predictLabel.append(0)
    return predictLabel

# Calculate the error:
def compute_error(y_pred, y):
    errorLabel=0
    sum=y.shape[0]
    for i in range(y.shape[0]):
        if y_pred[i]!=y[i]:
            errorLabel+=1
    error=errorLabel/sum
    return error

if __name__=='__main__':
    trainInput=sys.argv[1]
    validationInput=sys.argv[2]
    testInput=sys.argv[3]
    trainOut=sys.argv[4]
    testOut=sys.argv[5]
    metricsOut=sys.argv[6]
    num_epoch=sys.argv[7]
    learning_rate=sys.argv[8]
    # model2inputTrain=sys.argv[9]
    # model2inputValid=sys.argv[10]
    # model3input=sys.argv[11]

    #X,y=getLabelAndFeature(trainInput)
    #X3,y3=getLabelAndFeature(testInput)
    #theta=np.zeros((X.shape[1],))
    epoch=int(num_epoch)
    rate=float(learning_rate)
    #theta, trainLikelihood=train(theta, X, y, epoch, rate)
    #trainLabel=predict(theta, X)
    #testLabel=predict(theta, X1)
    #errorTrain=compute_error(trainLabel, y)
    #errorTest=compute_error(testLabel, y1)
    X1,y1=getLabelAndFeature(trainInput)
    X2,y2=getLabelAndFeature(validationInput)
    theta=np.zeros((X1.shape[1],))
    # thetasmall=np.zeros((X1.shape[1],))
    # thetareallysmall=np.zeros((X1.shape[1],))
    #X2,y2=getLabelAndFeature(validationInput)
    theta1, t1, t2=train(theta, X1, y1, epoch, rate, X2, y2) 
    # theta2, t2=train(thetasmall, X1, y1, epoch, 0.00001) 
    # theta3, t3=train(thetareallysmall, X1, y1, epoch, 0.000001) 
    # trainLabel=predict(theta1, X1)
    # testLabel=predict(theta1, X3)
    # errorTrain=compute_error(trainLabel, y1)
    # errorTest=compute_error(testLabel, y3)
    #with open(trainOut, 'w') as file:
    #    for result in trainLabel:
    #        file.write(str(result) + '\n')

    #with open(testOut, 'w') as file:
    #    for result in testLabel:
    #        file.write(str(result) + '\n')

    # with open(metricsOut, 'w') as file:
    #     file.write('error(train): ' + str(format(errorTrain,'.6f')) + '\n' + 'error(test): ' + str(format(errorTest,'.6f')))
    
    #thetaV=np.zeros((X2.shape[1],))
    #thetaV,validLikelihood=train(thetaV, X2, y2, epoch, rate)

    epochX=list(range(1,51))

    # X3,y3=getLabelAndFeature(model2inputTrain)
    # theta2=np.zeros((X3.shape[1],))
    # X4,y4=getLabelAndFeature(model2inputValid)
    # theta2, tmodel2, t2=train(theta2, X3, y3, epoch, rate, X4, y4) 
    # X5,y5=ge(model3input) 
    # t3=X5[:,-1]

    # plt.plot(epochX, t1, label='train')
    # plt.plot(epochX, t1, label='Model1')
    # plt.plot(epochX, t2, label='Model2')
    # plt.plot(epochX, t3, label='Model3')
    plt.plot(epochX, t1, label='train')
    plt.plot(epochX, t2, label='validation')
    # plt.plot(epochX, t3, label='alpha=10^-6')
    # plt.plot(epochX, t2, label='validation')
    plt.xlabel('epoch')
    plt.ylabel('negative log likelihood')
    plt.legend(loc='upper right')
    plt.show()
