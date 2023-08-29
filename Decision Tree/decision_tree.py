import numpy as np
import sys, math
import matplotlib.pyplot as plt

np.random.seed(0)

class Node:
    def __init__(self, feature, attrName, depth, max_depth, labelList, indexList=[], left=None, right=None):
        self.left=left
        self.right=right
        self.attr=None
        self.depth=depth
        self.max_depth=max_depth
        self.indexList=indexList

        # the attributes & label of attributes (aka. features) of nodes from the dataset:
        self.feature=feature
        self.attrName=attrName # without the Label of result column

        # parameters used for recursion:
        self.maxMInfo=0
        self.splitAttributeIndex=None
        self.splitAttribute=None  # record the name of the attribute

        self.leftAttr=None
        self.rightAttr=None

        self.labelList=labelList

    # Help function1: majority vote:
    def majorityVote(self, data):
        label=data[:,-1]
        if len(label)==0: self.predict, self.predictCount=None,-1
        else:
            firstLabel=label[0]
            firstCount,secondCount=0,0
            for l in label:
                if l==firstLabel:
                    firstCount+=1
                else:
                    secondLabel=l
                    secondCount+=1
            if secondCount==0:
                for l in self.labelList:
                    if l!=firstLabel:
                        secondLabel=l
            if firstCount>secondCount:
                self.predict=firstLabel
                self.predictCount=firstCount
                self.noPredict=secondLabel
                self.noPreCount=secondCount
            elif firstCount<secondCount:
                self.predict=secondLabel
                self.predictCount=secondCount
                self.noPredict=firstLabel
                self.noPreCount=firstCount                
            else:
                if firstLabel<secondLabel:
                    self.predict=secondLabel
                    self.predictCount=secondCount
                    self.noPredict=firstLabel
                    self.noPreCount=firstCount  
                else:
                    self.predict=firstLabel
                    self.predictCount=firstCount
                    self.noPredict=secondLabel
                    self.noPreCount=secondCount      
        return self.predict, self.predictCount

    # Help function2: calculate the entropy:
    def calEntropy(self, data):
        label=data[:,-1]
        sum=len(label)
        predictLabel,predictCount=self.majorityVote(data)
        if sum==0 or predictCount==sum: 
            p, self.entropy=0, 0
        else:
            p=predictCount/sum
            self.entropy=-p*math.log2(p)-(1-p)*math.log2(1-p)
        return self.entropy
    
    # Help function3: divide the label dataset based on the chosen attribute in order to calculate entropy:
    def divide(self, attributeIndex, data):
        attr1=data[0,attributeIndex]  # the first value of attribute: go to left node
        data1=data[data[:,attributeIndex]==attr1]
        data2=data[data[:,attributeIndex]!=attr1]
        self.leftAttr=attr1
        p1=len(data1[:,0])/len(data[:,0])
        if len(data2[:,0])!=0:
            self.rightAttr=data2[0,attributeIndex]
            p2=len(data2[:,0])/len(data[:,0])
        else:
            self.rightAttr=None
            p2=0
        return p1, p2, data1, data2
    
    # Help function4: calculate the mutual information:
    def mutualInformation(self, attributeIndex, data):
        p1, p2, label1, label2=self.divide(attributeIndex, data)
        entropy1=self.calEntropy(label1)
        entropy2=self.calEntropy(label2)
        totalEntropy=self.calEntropy(data)
        mInfo=totalEntropy-p1*entropy1-p2*entropy2
        return mInfo

    # train nodes:
    def train(self):
        self.entropy=self.calEntropy(self.feature)
        if self.depth > self.max_depth-1: # stop when meet the max depth
            self.attr='leaf'
            return
        elif len(self.indexList)==len(self.attrName): # stop when no more attributes can be splitted
            self.attr='leaf'
            return
        elif self.entropy==0: # stop when entropy is 0: no child nodes
            self.attr='leaf'
            return
        else:
            for i in range(len(self.attrName)):
                if i not in self.indexList:
                    mInfo=self.mutualInformation(i, self.feature)
                    if mInfo<self.maxMInfo or mInfo==self.maxMInfo: continue
                    elif mInfo>self.maxMInfo:
                        self.maxMInfo, self.splitAttributeIndex, self.splitAttribute=mInfo, i, self.attrName[i]
            if self.maxMInfo==0:  # stop when the mutual information is 0
                self.attr='leaf'
                return
            self.indexList.append(self.splitAttributeIndex)
            newFeature, newAttrName=self.feature, self.attrName
            temp1,temp2,leftFeature, rightFeature=self.divide(self.splitAttributeIndex, newFeature)
            leftAttrName, rightAttrName=newAttrName, newAttrName
            self.attr='inter'
            self.left=Node(leftFeature, leftAttrName, self.depth+1, self.max_depth, self.labelList, self.indexList.copy(), left=None, right=None)
            self.right=Node(rightFeature, rightAttrName, self.depth+1, self.max_depth, self.labelList, self.indexList.copy(), left=None, right=None)
            self.left.train()
            self.right.train()
            return

    def prediction(self, row):
        if self.left==None and self.right==None:
            return self.predict
        else:
            if row[self.splitAttributeIndex]==self.leftAttr:
                return self.left.prediction(row)
            else: return self.right.prediction(row)


def test(file, node):
    result=[]
    data=np.genfromtxt(file, delimiter='\t', dtype=None, encoding=None)
    feature=data[1:,:]
    attrName=data[0,:-1]
    for i in range(len(feature[:,-1])):
        result.append(node.prediction(feature[i]))
    return result

def readTrainData(file):
    data=np.genfromtxt(file, delimiter='\t', dtype=None, encoding=None)
    feature=data[1:,:]
    attrName=data[0,:-1]
    labelValue1=feature[0,-1]
    for i in range(len(feature[:,-1])):
        if feature[i,-1]!=labelValue1:
            labelValue2=feature[i,-1]
    labelList=[labelValue1, labelValue2]
    return feature,attrName, labelList

def printTree(node):
    if node:
        attr=node.splitAttribute
        attrIndex=node.splitAttributeIndex
        attrV1, attrV2=node.leftAttr, node.rightAttr
        
        if node.left!=None:
            for i in range(node.depth+1):
                print('|', end='')
            print(attr, ' = ', attrV1, ': ','[', node.left.predictCount, node.left.predict, '/', node.left.noPreCount, node.left.noPredict,']')
            printTree(node.left)
        
        if node.right!=None:
            for i in range(node.depth+1):
                print('|', end='')
            print(attr, ' = ', attrV2, ': ','[', node.right.predictCount, node.right.predict, '/', node.right.noPreCount, node.right.noPredict,']')
            printTree(node.right)

def printResult(node):
    print('[', node.predictCount, node.predict, '/', node.noPreCount, node.noPredict,']', end='\n')
    printTree(node)

def calErrorRate(predict,file):
    dataset=np.genfromtxt(file, delimiter='\t', dtype=None, encoding=None)
    data=dataset[1:,]
    label=data[:,-1]
    errorCount=0
    sum=len(label)
    for i in range(sum):
        if predict[i]!=label[i]:
            errorCount+=1
    return errorCount/sum



if __name__ == '__main__':
    trainInput=sys.argv[1]
    testInput=sys.argv[2]
    maxDepth=sys.argv[3]
    trainOut=sys.argv[4]
    testOut=sys.argv[5]
    metricsOut=sys.argv[6]

    feature, attrName, labelList=readTrainData(trainInput)
    root=Node(feature, attrName, 0, int(maxDepth), labelList, indexList=[], left=None, right=None)
    root.train()
    trainResult=test(trainInput, root)
    testResult=test(testInput, root)
    errorRateTrain=calErrorRate(trainResult, trainInput)
    errorRateTest=calErrorRate(testResult, testInput)

    with open(trainOut, 'w') as file:
        for result in trainResult:
            file.write(result + '\n')

    with open(testOut, 'w') as file:
        for result in testResult:
            file.write(result + '\n')

    with open(metricsOut, 'w') as file:
        file.write('error(train): ' + str(errorRateTrain) + '\n' + 'error(test): ' + str(errorRateTest))

    printResult(root)

    # feature, attrName, labelList=readTrainData('D:\politicians_train.tsv')
    # maxDepthOfData=len(attrName)
    # traine=[]
    # teste=[]
    # x=[]
    # for i in range(maxDepthOfData+1):
    #     root=Node(feature, attrName, 0, int(i), labelList, indexList=[], left=None, right=None)
    #     root.train()      
    #     trainResult=test('D:\politicians_train.tsv', root)
    #     testResult=test('D:\politicians_test.tsv', root)
    #     errorRateTrain=calErrorRate(trainResult, 'D:\politicians_train.tsv')  
    #     errorRateTest=calErrorRate(testResult, 'D:\politicians_test.tsv')  
    #     traine.append(errorRateTrain)
    #     teste.append(errorRateTest)
    #     x.append(i)
    
    # xv=x
    # t1=traine
    # t2=teste
    # plt.xlabel('max-depth')
    # plt.ylabel('error rate')
    # plt.plot(xv,t1, label='training error')
    # plt.plot(xv,t2, label='testing error')
    # plt.legend(loc = 'upper right')
    # for a,b in zip(xv,t1):
    #     plt.text(a,b,round(b,4),ha='center',va='bottom',fontsize=10)
    # for a,b in zip(xv,t2):
    #     plt.text(a,b,round(b,4),ha='center',va='bottom',fontsize=10)    
    # plt.show()    
