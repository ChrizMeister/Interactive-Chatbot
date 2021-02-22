# Christopher Sommerville - CS 175 Winter 2020
# Adapted from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

import torch, os
from gtts import gTTS
from playsound import playsound

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

from Classes import GreedyDecoder
from Processing import process_data, normalize_string
from Training import train_model, evaluate


def run_model(encoder, decoder, searcher, vocab, max_length):
    input_sentence = ''
    while(True):
        try:
            input_sentence = input('You: ')
            if input_sentence == 'quit': 
                break

            input_sentence = normalize_string(input_sentence)
            output_words = evaluate(encoder, decoder, searcher, vocab, \
                                    input_sentence, max_length)

            index = 0
            for i in range(len(output_words)):
                if output_words[i] == 'EOS':
                    index = i
                    break
            output_words = output_words[:index + 1] # Cutoff words past EOS
            output_words[:] = [word for word in output_words if not \
                               (word == 'EOS' or word == 'PAD')]
                    
            output = ' '.join(output_words)
            
            response = ''
            
            eos_tokens = ['.', '?', '!']
            
            # Clean up some of the output formatting
            for i in range(len(output)):
                if output[i] == ' ':
                    if i < len(output) - 1:
                        if output[i + 1] not in eos_tokens:
                            response += ' '
                else:
                    if i == 0:
                        response += output[i].upper()
                    else:
                        response += output[i]
                    
                
            print('Bot:', response)
            
            play_audio_response(response)

        # If the word is not found in the vocab object, raise an error
        except KeyError:
            print("Sorry I don't know that word")

def play_audio_response(response):
    if os.path.exists("response.mp3"):
        os.remove("response.mp3")
    tts = gTTS(text=response, lang='en')
    tts.save("response.mp3")
    playsound('response.mp3')


def main():
    input_file = 'data/train.txt'
    sample_size = 1.0
    outfile_name = 'data.txt'

    # Set to true if running for the first time
    create_output = False
    
    # Set load to false if running for the first time
    load = True
    
    max_length = 15 # Max length of sentences to consider
    min_count = 3   # Minimum amount of occurunces for rare words
    
    vocab, pairs = process_data(input_file, sample_size, outfile_name,
                                create_output, load, max_length, min_count)
    
    # Model and Training parameters
    save_dir = 'models'    
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64
    hidden_size = 400
    clip = 50.0
    learning_rate = 0.0005
    decoder_learning_ratio = 5.0
    n_iteration = 5000
    print_every = 10
    save_every = n_iteration # lower if model cannot be trained in one session

    # Set load_file = None if training from scratch
    load_file = os.path.join(save_dir,'testmodel.tar')
    # load_file = None
    save_name = 'model'
    encoder, decoder = \
        train_model(vocab, pairs, save_dir, encoder_n_layers, decoder_n_layers,
                    dropout, batch_size, hidden_size, clip, learning_rate,
                    decoder_learning_ratio, n_iteration, print_every, save_every,
                    max_length, load_file, save_name)

    searcher = GreedyDecoder(encoder, decoder)

    
    run_model(encoder, decoder, searcher, vocab, max_length)
    # remove last response
    if os.path.exists("response.mp3"):
        os.remove("response.mp3")

if __name__ == '__main__':
    main()