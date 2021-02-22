# Christopher Sommerville - CS 175 Winter 2020
# Adapted from https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

import itertools
import random
import os
import torch
import torch.nn as nn
from torch import optim

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

from Classes import SOS_token, EOS_token, PAD_token
from Classes import Encoder, Decoder

# Some alterations from base code
def train_model(vocab, pairs, save_dir = 'models', encoder_n_layers = 2, decoder_n_layers = 2,
                dropout = 0.1, batch_size = 30, hidden_size = 500, clip = 50.0, learning_rate = 0.0001,
                decoder_learning_ratio = 5.0, n_iteration = 3000, print_every = 30, save_every = 1000,
                max_length = 12, load_file = None, out_name = 'model'):
    checkpoint = None
    if load_file:
        checkpoint, encoder_sd, decoder_sd, encoder_optimizer_sd, decoder_optimizer_sd, \
        embedding_sd, vocab.__dict__ = load_checkpoint(load_file)
        
    embedding = nn.Embedding(vocab.num_words, hidden_size)
    if load_file:
        embedding.load_state_dict(embedding_sd)
        
    encoder = Encoder(hidden_size, embedding, encoder_n_layers, dropout).to(device)
    decoder = Decoder('general', embedding, hidden_size, vocab.num_words, decoder_n_layers, dropout).to(device)
    
    if load_file:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
        
    encoder.train()
    decoder.train()
       
    # Load/Build optimizers
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if load_file:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)
    
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    
    for state in decoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    
    # Run training if not loading a pretrained model
    if not load_file:
        trainIters(vocab, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                   embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
                   print_every, save_every, clip, checkpoint, hidden_size, max_length, load_file, out_name)
    else:
        print("Loading pretrained model")
        print("Type 'quit' to exit")
        
    return encoder, decoder
    
def load_checkpoint(load_file):
    checkpoint = torch.load(load_file)  
    return checkpoint, checkpoint['en'], checkpoint['de'], checkpoint['en_opt'], \
    checkpoint['de_opt'], checkpoint['embedding'], checkpoint['voc_dict']

# Unchanged from base code
def indexes_from_sentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# Unchanged from base code
def zero_padding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# Unchanged from base code
def binary_matrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Unchanged from base code
# Returns padded input sequence tensor and lengths
def input_var(l, vocab):
    indexes_batch = [indexes_from_sentence(vocab, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths


# Unchanged from base code
# Returns padded target sequence tensor, padding mask, and max target length
def output_var(l, vocab):
    indexes_batch = [indexes_from_sentence(vocab, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    mask = binary_matrix(pad_list)
    mask = torch.BoolTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len

# Unchanged from base code
# Returns all items for a given batch of pairs
def batch_to_train_data(vocab, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, vocab)
    output, mask, max_target_len = output_var(output_batch, vocab)
    return inp, lengths, output, mask, max_target_len


# Unchanged from base code
def masked_loss(inp, target, mask):
    n_total = mask.sum()
    cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = cross_entropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, n_total.item()

# Mostly unchanged from base code
# Run a single training interation
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for x in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Forward batch of sequences through decoder one time step at a time
    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

        decoder_input = target_variable[t].view(1, -1)

        mask_loss, nTotal = masked_loss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

# Mostly unchanged from base code
def trainIters(vocab, pairs, encoder, decoder, encoder_optimizer, 
               decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, 
               save_dir, n_iteration, batch_size, print_every, save_every, clip, 
               checkpoint, hidden_size, max_length, load_file, out_name):
    # Create the batches to be used in each training iteration
    # Each batch uses a random sample of the dataset and the entire vocabulary
    training_batches = \
    [batch_to_train_data(vocab, [random.choice(pairs) for x in range(batch_size)])
     for x in range(n_iteration)]

    # Initializations
    start_iteration = 1
    print_loss = 0
    if load_file:
        print("Loading checkpoint")
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    if start_iteration < n_iteration:
        print("Starting training")

    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length)
        print_loss += loss
            
        # Output current iteration # and average loss so far
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".\
                  format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        file_output = out_name + '.tar'
        
        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': vocab.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, file_output))
            
# Unchanged from base code
def evaluate(encoder, decoder, searcher, vocab, sentence, max_length):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexes_from_sentence(vocab, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [vocab.index2word[token.item()] for token in tokens]
    return decoded_words