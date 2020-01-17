#!python3

'''

create models

train models

'''

import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, CuDNNLSTM, Dropout, Embedding

import Custom_Translator_Data_Management as cust_data

'''
def create_encoder_OLD(vocab_size, embedding_dim, encoding_units):
    # Basic Structure:
    # LSTM input layer
    # middle layers (if required)
    # Dense Output layer of embedding size
    encoder = Sequential()

    encoder.add(Embedding(vocab_size, embedding_dim))

    encoder.add(CuDNNLSTM(encoding_units)) # add in input_shape?

    opt = keras.optimizers.Adam()

    # figure out a loss func
    encoder.compile(loss=tf.nn.sparse_softmax_cross_entropy_with_logits(),
                    optimizer=opt)


def create_decoder_OLD():
    decoder = Sequential()
'''

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoding_dim, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoding_dim = encoding_dim
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(dec_units, 
                       return_sequences=True, 
                       return_state=True, 
                       recurrent_activation='sigmoid', 
                       recurrent_initializer='glorot_uniform')

    def call(self, x, gru_init_state):
        x = self.embedding(x)
        output, state = self.gru(x, intial_state=gru_init_state)
        return output, state

    def intialize_gru_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))


def train():
    pass


# train model








