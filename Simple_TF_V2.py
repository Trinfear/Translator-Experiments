#!python3

from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import time

from tensorflow.keras import Model
from keras.layers import Embedding, GRU, Dense, CuDNNGRU
from tensorflow.nn import tanh, softmax
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits as sparse_logits
from tensorflow.keras.preprocessing.sequence import pad_sequences

# TODO: add in function to check for running gru on gpu, so desktop is faster?
# TODO: add in a tensorboard?
# TODO: add in detailed decription of dataset shape and formation for training?



# define global variables here
    # should any of of these not just be in if __main__ statement?



# sentence processing functions

def unicode_to_ascii(sentence):
    return ''.join(c for c in unicodedata.normalize('NFD', sentence)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sentence):
    sentence = unicode_to_ascii(sentence.lower().strip())

    # pad punctuation with white spaces, so they are their own words when split
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # clean out other characters, replace with spaces
    sentence = re.sub(r"^a-zA-Z?.!,¿]+", " " , sentence)
    sentence = sentence.rstrip().strip()
    sentence = '<start> ' + sentence + ' <end>'

    return sentence


# create models
class Encoder(Model):
    # TODO: add in functionality for adding new words and stuff?
    def __init__(self, vocab_size, embed_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embed_dim)
        self.gru = GRU(enc_units, return_sequences=True, return_state=True,         # try switching this to CuDNN?
                            recurrent_activation='sigmoid',
                            recurrent_initializer='glorot_uniform')

    def hidden_state_init(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state


class Decoder(Model):
    def __init__(self, vocab_size, embed_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embed_dim)
        self.gru = GRU(dec_units, return_sequences=True,return_state=True,         # try switching this to CuDNN?
                            recurrent_activation='sigmoid',
                            recurrent_initializer='glorot_uniform')
        self.final_output = Dense(vocab_size)

        # weights for attention
        self.W1 = Dense(self.dec_units)
        self.W2 = Dense(self.dec_units)
        self.Verdict = Dense(1)

    def hidden_state_init(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def call(self, x, hidden, enc_output):
        # Attention
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        score = self.Verdict(tanh(self.W1(enc_output) +
                                  self.W2(hidden_with_time_axis)))

        attention_weights = softmax(score, axis=1)

        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # forward pass
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.final_output(output)

        return x, state, attention_weights

    '''
    shape on pass through Decoder Call func:
    
    inputs:
    
        enc_output_shape == (batch_size, max_length, hidden_size)
        hidden_shape == (batch_size, max_length, 1)
        hidden_with_time_axis_shape = (batch_size, 1, hidden_size)

    Attention:

        score_shape == (batch_size, max_length, 1)
        attention_weights_after_sum_shape == (batch_size, max_length, 1)
        context_vector_after_sum_shape == (batch_size, hidden_size)

    Forward Pass:

        x_after_embedding_shape == (batch_size, 1, embed_dim)
        x_after_concat_shape == (batch_size, 1, embed_dim + hidden_size)
        output_shape == (batch_size, hidden_size)

    '''

class Translator():
    def __init__(encoder, decoder):
        pass
    # create a class which contains the encoder and decoder for two langs
        # and funcs to for translation and training

    # can then either create a large number of encoders/decoders or can create
        # a translator for a specific language set

    # basically just stick this whole script into a class


# create training and predicting funcs


def train_networks(enc, dec, epochs, dataset, optimizer, inp_lang, targ_lang,
                   batch_size, N_BATCH, check=None, check_prefix=None):

    # TODO:  make sure this is clearing data properly?
    # seems to generate an error at third epoch, even across multiple models
        # (if models train on less than three epochs)
    # perhaps it's not clearing the cache correctly?
    '''

    enc = encoder_model
    dec = decoder_model
    epochs = epochs to train
    dataset = training data, generated from tf.data.Dataset.from_tensor_slices
        tensor_slices = input_tensor_x and target_tensor_x
    
    optimizer = optimizer function
    inp_lang = language index for encoder model
    targ_lang = language index for decoder model
    check = tf.train.checkpoint
    check_prefix = file location for checkpoint saves

    '''

    # does having this here, as well as in master_translator actually matter?
    # should it just be in one locaiton?
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    print('Epochs: ', epochs)

    def custom_loss(real, pred):
        mask = 1 - np.equal(real, 0)
        loss_value = sparse_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_value)

    
    for epoch in range(epochs):
        start = time.time()
        
        hidden = enc.hidden_state_init()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0

            with tf.GradientTape() as tape:

                # run forward pass
                enc_output, enc_hidden = enc(inp, hidden)

                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([targ_lang.word_to_id['<start>']] * batch_size, 1)

                for t in range(1, targ.shape[1]):
                    predictions, dec_hidden, atten_weights = dec(dec_input,
                                                                 dec_hidden,
                                                                 enc_output)

                    loss += custom_loss(targ[:, t], predictions)
                    
                    dec_input = tf.expand_dims(targ[:, t], 1)

                # train networks
                
                batch_loss = (loss / int(targ.shape[1]))
                total_loss += batch_loss

                variables = enc.variables + dec.variables
                gradients = tape.gradient(loss, variables)

                optimizer.apply_gradients(zip(gradients, variables))

                if batch % 100 == 0:
                    print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch+1,
                                                                    batch,
                                                                    batch_loss.numpy()))
        if check is not None and (epoch + 1) % 2 == 0:
            check.save(file_prefix = check_prefix)

        print('Epoch: {}, Loss {:.4f}'.format(epoch + 1,
                                              total_loss / N_BATCH))

        print('Time taken for 1 epoch: {} seconds\n'.format(time.time() - start))

    return enc, dec


def translate(sentence, enc, dec, inp_lang, targ_lang, units, max_input_length=25,
              max_targ_length=25):
    '''

    sentence = sentence to be translated
    enc = encoder model to be used
    dec = decoder model to be used
    inp_lang = language index for encoder
    targ_lang = language index for decoder
    
    max_input_length = maximum length of input sentences
    max_targ_length = maximum length of output sentences

    units = encoder/decoder units?

    '''

    attention_plot = np.zeros((max_targ_length, max_input_length))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_to_id[i] for i in sentence.split(' ')]
    inputs = pad_sequences([Inputs], maxlen=max_input_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = '<start> '

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = enc(inputs_hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_to_id['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, atten_weights = dec(dec_input, dec_hidden, enc_out)

        # storing attention weights for graphing later
        atten_weights = tf.reshape(atten_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        # getting prediction
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = targ_lang.id_to_word[predicted_id]

        result +=  predicted_word + ' '

        if predicted_word == '<end>':
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)

    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()


def test():
    # run through translate and print input and predictions in console
    pass



if __name__ == '__main__':
    # load data
    # generate models
    # train models
    # run some predictions
    pass







