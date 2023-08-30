############ Welcome to HW7 ############
#Andrew-id: zehanz


# Imports
# Don't import any other library
import argparse
import numpy as np
from utils import make_dict, parse_file
import logging

# Setting up the argument parser
# don't change anything here

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .txt file')
parser.add_argument('index_to_word', type=str,
                    help='path to index_to_word.txt file')
parser.add_argument('index_to_tag', type=str,
                    help='path to index_to_tag.txt file')
parser.add_argument('init', type=str,
                    help='path to store the hmm_init.txt (pi) file')
parser.add_argument('emission', type=str,
                    help='path to store the hmm_emission.txt (A) file')
parser.add_argument('transition', type=str,
                    help='path to store the hmm_transition.txt (B) file')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')

def cal(transition, emission, init, word_dict, tag_dict, sentences, tags):
    for m in range(len(tags)):
        line=tags[m]
        init[tag_dict[line[0]], 0]+=1
        for i in range(len(line)):
            if i < len(line)-1:
                transition[tag_dict[line[i]], tag_dict[line[i+1]]]+=1
            emission[tag_dict[line[i]], word_dict[sentences[m][i]]]+=1
    transition=transition/np.sum(transition, axis=1, keepdims=True)
    emission=emission/np.sum(emission, axis=1, keepdims=True)
    init=init/np.sum(init)
    return init, transition, emission

# Complete the main function
def main(args):

    # Get the dictionaries
    word_dict = make_dict(args.index_to_word)
    tag_dict = make_dict(args.index_to_tag)

    sentences, tags = parse_file(args.train_input)

    logging.debug(f"Num Sentences: {len(sentences)}")
    logging.debug(f"Num Tags: {len(tags)}")
    
    
    # Train HMM
    init = np.ones((len(tag_dict),1))
    emission = np.ones((len(tag_dict),len(word_dict)))
    transition = np.ones((len(tag_dict),len(tag_dict)))

    init, transition, emission=cal(transition, emission, init, word_dict, tag_dict, sentences[0:10000], tags[0:10000])

    # Making sure we have the right shapes
    logging.debug(f"init matrix shape: {init.shape}")
    logging.debug(f"emission matrix shape: {emission.shape}")
    logging.debug(f"transition matrix shape: {transition.shape}")
    
    np.savetxt(args.init, init)
    np.savetxt(args.emission, emission)
    np.savetxt(args.transition, transition)

    return 

if __name__ == "__main__":
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    main(args)