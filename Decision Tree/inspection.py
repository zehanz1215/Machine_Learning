import numpy as np
import sys, math

np.random.seed(0)

# Train model and get the result of the majority vote classifier, the count of the majority vote, and the total count:
def majorityVote(file):
    dataset=np.genfromtxt(file, delimiter='\t', dtype=None, encoding=None)
    data=dataset[1:,]
    label=data[:,-1]
    count=len(label)
    firstLabel=label[0]
    firstLabelCount, secondLabelCount=0, 0
    for l in label:
        if l==firstLabel:
            firstLabelCount+=1
        else:
            secondLabel=l
            secondLabelCount+=1
    if firstLabelCount > secondLabelCount:
        predictCount=firstLabelCount
    elif firstLabelCount < secondLabelCount:
        predictCount=secondLabelCount
    else:
        if firstLabel < secondLabel:
            predictCount=secondLabelCount
        else:
            predictCount=firstLabelCount
    return predictCount,count

# Calculate the entropy and the error rate:
def errorRateAndEntropy(predictCount, count):
    errorRate=(count-predictCount)/count
    p=predictCount/count
    entropy=-p*math.log2(p)-(1-p)*math.log2(1-p)
    return entropy, errorRate

if __name__ == '__main__':
    input=sys.argv[1]
    output=sys.argv[2]

    predictCount, count=majorityVote(input)
    entropy, errorRate=errorRateAndEntropy(predictCount, count)

    with open(output, 'w') as file:
        file.write('entropy: '+str(entropy) +'\n'+ 'error: ' + str(errorRate))