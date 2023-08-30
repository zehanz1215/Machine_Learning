############ Welcome to HW7 ############
# Andrew-id: zehanz


# Imports
# Don't import any other library
import numpy as np
from utils import make_dict, parse_file, get_matrices, write_predictions, write_metrics
import argparse
import logging

# Setting up the argument parser
# don't change anything here
parser = argparse.ArgumentParser()
parser.add_argument('validation_input', type=str,
                    help='path to validation input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to the learned hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to the learned hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to the learned hmm_transition.txt (B) file')
parser.add_argument('prediction_file', type=str,
                    help='path to store predictions')
parser.add_argument('metric_file', type=str,
                    help='path to store metrics')                    
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')


def alphaAndBeta(tag_dict, word_dict, init, emission, transition, sentence):
    alpha=np.zeros((len(tag_dict),len(sentence)))
    beta=np.zeros((len(tag_dict),len(sentence)))
    log_trans=np.log(transition)
    log_emission=np.log(emission)
    # calculate alpha:
    alpha[:,0]=np.log(np.multiply(init, emission[:, word_dict[sentence[0]]]))
    print(alpha)
    for t in range(1, len(sentence)):
        alpha_t=log_trans+alpha[:,t-1].reshape((alpha.shape[0],1))
        max_alpha=np.max(alpha_t, axis=1)
        alpha[:,t]=max_alpha+np.log(np.sum(np.exp(alpha_t-max_alpha), axis=0))+log_emission[:, word_dict[sentence[t]]]
        print('t',alpha)
    # calculate beta:
    for j in range(len(sentence)-2, -1, -1):
        beta_j=(log_trans.T)+log_emission[:, word_dict[sentence[j+1]]].reshape((beta.shape[0],1))+beta[:,j+1].reshape((beta.shape[0],1))
        max_beta=np.max(beta_j, axis=1)
        beta[:,j]=max_beta+np.log(np.sum(np.exp(beta_j-max_beta), axis=0))
    return alpha, beta

def predict(sentences, tag_dict, word_dict, init, emission, transition):
    predicted_tags=[]
    log_likelihood=0
    for sentence in sentences:
        predicted_tag=[]
        alpha, beta=alphaAndBeta(tag_dict, word_dict, init, emission, transition, sentence)
        print(alpha)
        # find predicted_tag:
        probability=alpha+beta
        index=np.argmax(probability, axis=0)
        dict_tag={v: k for k, v in tag_dict.items()}
        for i in index:
            predicted_tag.append(dict_tag[i])
        predicted_tags.append(predicted_tag)
        # calculate log_likelihood:
        max_likelihood=np.max(alpha[:,-1])
        log_likelihood+=max_likelihood+np.log(np.sum(np.exp(alpha[:,-1]-max_likelihood)))
    avg_log_likelihood=log_likelihood/len(sentences)
    return predicted_tags, avg_log_likelihood

# Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    # Parse the validation file
    sentences, tags = parse_file(args.validation_input)

    init, emission, transition = get_matrices(args)

    predicted_tags, avg_log_likelihood=predict(sentences, tag_dict, word_dict, init, emission, transition) 
    accuracy = 0 # We'll calculate this for you

    accuracy = write_predictions(args.prediction_file, sentences, predicted_tags, tags)
    write_metrics(args.metric_file, avg_log_likelihood, accuracy)

    return

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)