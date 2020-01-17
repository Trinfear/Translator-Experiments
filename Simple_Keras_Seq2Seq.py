#!python3

'''

word based translator based on keras tutorial

intake data from other custom seq2seq data cleaner
create a model based on keras tutorial
train model based on keras tutorial

create a predict func

save model

'''

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, CuDNNLSTM, Dense, Embedding
import numpy as np


def load_data():
    pass

'''
def create_model(encoder_token_count, decoder_token_count, encoding_dim, embedding_dim):

    encoder_inputs = Input(shape=(None,))
    x = Embedding(encoder_token_count, encoding_dim)(encoder_inputs)
    x, state_h, state_c = CuDNNLSTM(encoding_dim, return_state=True)(x)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    x = Embedding(decoder_token_count, embedding_dim)(decoder_inputs)
    x = CuDNNLSTM(embedding_dim,
                  return_sequences=True)(x, initial_state=encoder_states)

    decoder_outputs = Dense(decoder_token_count, activation='softmax')(x)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return model, encoder_inputs, decoder_inputs, decoder_outputs


def train_model(model, encoder_input_data, decoder_input_data,
                decoder_target_data, batch_size, epochs, validation_split):

    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split)
    pass
'''


class Translator:
    def __init__(self, encoder_token_count, decoder_token_count, encoding_dim, embedding_dim,
                 input_token_index, target_token_index):
        # creates the three important models:
        # self.training_model (trains weights for other models)
        # self.encoder_model (important in predict function
        # self.decoder_model (important in predict funciton)

        # creating values for training model

        encoder_inputs = Input(shape=(None,))
        
        x = Embedding(encoder_token_count, encoding_dim)(encoder_inputs)
        x, state_h, state_c = CuDNNLSTM(encoding_dim,
                                        return_state=True)(x)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        
        x = Embedding(decoder_token_count, embedding_dim)(decoder_inputs)
        decoder_lstm = CuDNNLSTM(embedding_dim, return_sequences=True)
        x = decoder_lstm(x, initial_state=encoder_states)

        decoder_dense = Dense(decoder_token_count, activation='softmax')
        decoder_outputs = decoder_dense(x)

        # creating model for use in training

        self.training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.training_model.compile(optimizer='rmsprop',
                                    loss='categorical_crossentropy')

        # creating models for use in prediction
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(encoding_dim,))
        decoder_state_input_c = Input(shape=(encoding_dim,))
        
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)

        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                                   [decoder_outputs] + decoder_states)

        # create index lookups for characters

        self.reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

    def train(self, encoder_input_data, decoder_input_data,
              decoder_target_data, batch_size, epochs, validation_split):
        
        self.training_model.fit([encoder_input_data, decoder_input_data],
                                decoder_target_data,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_split=validation_split)

    def predict(self, input_seq):

        states_value = self.encoder_model.predict(input_seq)
        pass





class Seq2Seq():
    # recreate translator class from above but with more organized init?
    def __init__(self):

        self.training_model = self.create_training_model()

        self.encoder_model, self.decoder_model = self.create_prediction_models()

        self.reverse_input_dict, self.reverse_char_dict = self.get_char_dicts()

    def create_training_model(self):
        pass

    def create_prediction_models(self):
        pass

    def get_char_dicts(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass













