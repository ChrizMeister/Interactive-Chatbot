# Christopher Sommerville - CS 175 Winter 2020
# Adapted from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html


import unicodedata
import re
import pickle

from Classes import Vocabulary, EOS_token


def process_data(infile_name, sample_size, outfile_name, create_output, 
                 load, max_length, min_count):
    if create_output:
        create_datafile(infile_name, sample_size, outfile_name)
        
    vocab, pairs = None, None
    
    if not load: # Create vocab and pairs objects
        vocab, pairs = get_vocab(outfile_name)
        pairs = filter_pairs(pairs, max_length)

        for pair in pairs:
            vocab.addSentence(pair[0])
            vocab.addSentence(pair[1])
            
        pairs = trim_rare_words(vocab, pairs, min_count)
        
        # Save vocab and pairs objects
        with open('cb.vocab', 'wb') as vocab_file:
            pickle.dump(vocab, vocab_file)
        
        with open('cb.pairs', 'wb') as pairs_file:
            pickle.dump(pairs, pairs_file)
            
    else: # Load vocab and pairs objects
        with open('cb.vocab', 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        with open('cb.pairs', 'rb') as pairs_file:
            pairs = pickle.load(pairs_file)
            
    return vocab, pairs
    

# Expects an input file in the following format:
# 1 "Person 1 input (a)" tab "Person 2 response (b)"
# 2 Person 1 response (c)" tab "Person 2 response (d)"
# ...
# n Person 1 input (y)" tab "Person 2 response (z)"
# 1 Person 1 response" tab "Person 2 response"
# ...

# Creates an output file with the following format:
# "Person 1 input (a)" tab "Person 2 response (b)"
# "Person 2 response (b)" tab "Person 1 response (c)"
# ...

def create_datafile(infile_name, sample_size, outfile_name):
    input_file = open(infile_name, 'r')
    contents = input_file.readlines()
    contents = contents[:int(len(contents) * sample_size)]
    out_file = open(outfile_name, "w")

    for i in range(len(contents)):
        line_text = contents[i][2:] # cutoff the line number
        line_num = int(contents[i][0])

        phrase_1 = line_text.split('\t')[0].strip()
        phrase_2 = line_text.split('\t')[1].strip()

        if line_num == 1:
            out_file.write(phrase_1)
            out_file.write('\t')
            out_file.write(phrase_2)
            out_file.write('\n')
            out_file.write(phrase_2)
            out_file.write('\t')

        if line_num > 1:
            out_file.write(phrase_1)
            out_file.write('\n')
            out_file.write(phrase_1)
            out_file.write('\t')
            out_file.write(phrase_2)
            out_file.write('\n')
            if i < len(contents) - 1 and int(contents[i + 1][0]) != 1:
                out_file.write(phrase_2)
                out_file.write('\t')

    out_file.close()
    
# Unchanged from base code
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

# Unchanged from base code
# Convert to lowercase and keep only letters
def normalize_string(string):
    s = unicode_to_ascii(string.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Remove inputs where there is no response
def remove_empty(pairs):
    temp = []
    for pair in pairs:
        if len(pair) == 2:
            temp.append(pair)
    return temp

# Mostly unchanged from base code
# Read data file and return the pairs of inputs/responses and Vocabulary object
def get_vocab(datafile):

    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    
    vocab = Vocabulary()
    return vocab, remove_empty(pairs)


# Mostly unchanged from base code
# Returns True if both sentences in a pair are under the threshold for the
# max_length
def filter_pair(pair, max_length):
    # Input sequences need to preserve the last word for EOS token
    return len(pair[0].split(' ')) < max_length and \
           len(pair[1].split(' ')) < max_length

# Mostly unchanged from base code
# Filter each pair of sentences
def filter_pairs(pairs, max_length=12):
    return [pair for pair in pairs if filter_pair(pair, max_length)]

# Unchanged from base code
# Trim words in the vocab object and the pairs
def trim_rare_words(vocab, pairs, min_count=3):
    vocab.trim(min_count)

    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in vocab.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in vocab.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    return keep_pairs

# Unchanged from base code
def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ')] + [EOS_token]
