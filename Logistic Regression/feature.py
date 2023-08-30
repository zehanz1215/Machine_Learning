import csv, sys
import numpy as np

VECTOR_LEN = 300   # Length of word2vec vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and word2vec.txt



def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An N x 2 np.ndarray. N is the number of data points in the tsv file. The
        first column contains the label integer (0 or 1), and the second column
        contains the movie review string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset

def load_dictionary(file):
    """
    Creates a python dict from the model 1 dictionary.

    Parameters:
        file (str): File path to the dictionary for model 1.

    Returns:
        A dictionary indexed by strings, returning an integer index.
    """
    dict_map = np.loadtxt(file, comments=None, encoding='utf-8',
                          dtype=f'U{MAX_WORD_LEN},l')
    return {word: index for word, index in dict_map}


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the word2vec
    embeddings.

    Parameters:
        file (str): File path to the word2vec embedding file for model 2.

    Returns:
        A dictionary indexed by words, returning the corresponding word2vec
        embedding np.ndarray.
    """
    word2vec_map = dict()
    with open(file) as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            word2vec_map[word] = np.array(embedding, dtype=float)
    return word2vec_map

# MOdel 1: Bag of Words, return both the label and the features:
def modelOne(dataset, dictionary):
    N=dataset.shape[0]
    M=len(dictionary)
    result=np.zeros((N,M+1))
    for i in range(N):
        result[i,0]=dataset[i][0]
        for word in dataset[i][-1].split(' '):
            if word in dictionary.keys():
                result[i,dictionary[word]+1]=1
    return result

# Model 2: Word Embeddings, return both the label and the features:
def modelTwo(dataset, dictionary):
    N=dataset.shape[0]
    M=VECTOR_LEN
    result=np.zeros((N,M+1))
    for i in range(N):
        result[i,0]=dataset[i][0]
        sum=0
        for word in dataset[i][-1].split(' '):
            if word in dictionary.keys():
                sum+=1
                result[i,1:]+=dictionary[word]
        result[i,1:]=result[i,1:]/sum
    return result

def featureOutput(file, flag, dictMap, word2VecMap, output):
    dataset=load_tsv_dataset(file)
    if flag=='1':
        dictMap=load_dictionary(dictMap)
        result=modelOne(dataset, dictMap)
        np.savetxt(output, result, fmt='%d', delimiter='\t', newline='\n')
    elif flag=='2':
        word2VecMap=load_feature_dictionary(word2VecMap)
        result=modelTwo(dataset, word2VecMap)
        np.savetxt(output, result, fmt='%1.6f', delimiter='\t', newline='\n')


if __name__=='__main__':
    trainInput=sys.argv[1]
    validationInput=sys.argv[2]
    testInput=sys.argv[3]
    dictInput=sys.argv[4]
    featureDictionaryInput=sys.argv[5]
    trainOut=sys.argv[6]
    validationOut=sys.argv[7]
    testOut=sys.argv[8]
    flag=sys.argv[9]

    featureOutput(trainInput, flag, dictInput, featureDictionaryInput, trainOut)
    featureOutput(validationInput, flag, dictInput, featureDictionaryInput, validationOut)
    featureOutput(testInput, flag, dictInput, featureDictionaryInput, testOut)