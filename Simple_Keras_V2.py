#!python3


from __future__ import print_function

import re
import numpy as np
import unicodedata
from keras.models import Model
from keras.layers import Input, CuDNNLSTM, Dense, Embedding, Lambda, LSTM

import tensorflow as tf


def unicode_to_ascii(sentence):
    return ''.join(c for c in unicodedata.normalize('NFD', sentence)
                   if unicodedata.category(c) != 'Mn')

def load_data(data_dir, max_phrase_count=None):
    # need a list of sets of cleaned sentences
    # need a set of all words in each word set

    # returns:
    #   cleaned_data --> data organized into phrases, split into languages, with padding and characters
    #   input_word_set  --> set of all unique input words
    #   target_word_set --> set of all unique target words

    data = open(data_dir, encoding='UTF-8').read().strip()

    # organize data into phrases and languages

    data = data.split('\n')

    new_data = []

    if max_phrase_count:
        for phrase in data[:max_phrase_count]:
            phrase = phrase.split('\t')
            new_data.append(phrase)
    else:
        for phrase in data:
            phrase = phrase.split('\t')
            new_data.append(phrase)

    data = new_data
    del(new_data)

    # remove problematic characters, add padding, pad punctuation

    cleaned_data = []
    cleaned_data.append(['<start> <end> <pad>', '<start> <end> <pad>'])
    for phrase_set in data:
        cleaned_phrases = []
        for phrase in phrase_set:
            phrase = unicode_to_ascii(phrase)

            phrase = re.sub(r"([?.!,¿])", r" \1 ", phrase)
            phrase = re.sub(r'[" "]+', " ", phrase)
            phrase = re.sub(r"^a-zA-Z?.!,¿]+", " ", phrase)

            phrase = phrase.rstrip().strip()

            phrase = '<start> ' + phrase + ' <end>'

            phrase = phrase.lower()

            # add padding to end of phrases?

            cleaned_phrases.append(phrase)
            
        cleaned_data.append(cleaned_phrases)

    # generate sets of unique words from data
    
    input_word_set = []
    target_word_set = []

    for phrase_set in cleaned_data:
        for word in phrase_set[0].split(' '):
            input_word_set.append(word)

        for word in phrase_set[1].split(' '):
            target_word_set.append(word)

    input_word_set = list(set(input_word_set))
    target_word_set = list(set(target_word_set))

    input_texts = [text[0] for text in cleaned_data]
    output_texts = [text[1] for text in cleaned_data]
    
    return cleaned_data, input_word_set, target_word_set, input_texts, output_texts


def create_word_indices(word_set):
    index = 0
    forward_index = {}
    reverse_index = {}

    for word in word_set:
        forward_index[word] = index
        reverse_index[index] = word
        index += 1

    return forward_index, reverse_index


def vectorize_data(data, max_seq_len, token_count, token_index):
    pass


def pad_sentences(data_set, max_length):
    # iterate through and add '<pad>' to end of sentences until they are all same length?
    pass


class Translator():
    def __init__(self, input_index, input_reverse_index, output_index, output_reverse_index, max_pred_len):
        # intake indices for vectorizing sentences

        self.input_index = input_index
        self.input_reverse_index = input_reverse_index

        self.output_index = output_index
        self.output_reverse_index = output_reverse_index

        self.max_pred_len = max_pred_len

    def create_models(self, encoding_dim, embedding_dim, enc_input, dec_input,
                      target_set, batch_size, epochs, val_split, in_toke_count,
                      out_toke_count):
        # intake embedding dim
        # intake encoder dim
        # intake input_word_count
        # intake target_word_count
        # intake training and test data

        # generate training model
        # train model

        # generate prediction models
        # return prediction models


        # organize run through:
        # TODO, it thinks input data should be 2D? Why?
        print('\n\n')

        # TODO:  track shape through model
        #           what should shape be?  sentence_length * embedding?   just embedding at the end?
        #           what is shape in this instance?

        print(enc_input.shape)
        encoder_inputs = Input(shape=enc_input.shape[1:])
        print(encoder_inputs.shape)
        x = Embedding(in_toke_count, encoding_dim)(encoder_inputs)
        print(x.shape)
        print('====================================================')
        # need to reshape embedding for LSTM, all actions need to be keras layers, hence using lambda
        x = Lambda(lambda x: x[0])(x)
        print(x.shape)
        x, state_h, state_c = CuDNNLSTM(encoding_dim, input_shape=x.shape[1:],
                                        return_state=True)(x)
        print(x.shape)

        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=dec_input.shape[1:])
        print('====================================================')
        print(decoder_inputs.shape)
        # shape exists here because of decoer inputs
        x = Embedding(out_toke_count, encoding_dim)(decoder_inputs)
        x = Lambda(lambda x: x[0])(x)

        print(x.shape)
        
        decoder_lstm = CuDNNLSTM(encoding_dim, return_sequences=True)
        # here is where shape should dissapear
        # just take last output?
        # change before it becomes input?
        x = decoder_lstm(x, initial_state=encoder_states)
        print(x.shape)

        # reshape x here
        # problem is it is trying to predict 8 words at once, should only be predicting one

        decoder_dense = Dense(out_toke_count, activation='softmax')
        decoder_outputs = decoder_dense(x)
        # get index of the max as output and generate from there?

        print(decoder_outputs.shape)
        print(target_set[1].shape)

        # train weights:

        training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        print('\n\nmodel declared')
        training_model.compile(optimizer='rmsprop',
                               loss='categorical_crossentropy')
        print('model compiled\n\n')

        training_model.fit([enc_input, dec_input], target_set,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_split=val_split)

        # create prediction models
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(encoding_dim,))
        decoder_state_input_c = Input(shape=(encoding_dim,))
        decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
                                                         initial_state=decoder_input_states)

        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_model = Model([decoder_inputs] + decoder_input_states,
                                   [decoder_outputs] + decoder_states)

    '''
    def create_models(self, encoding_dim, embedding_dim, enc_input, dec_input,
                  target_set, batch_size, epochs, val_split, in_toke_count,
                  out_toke_count):
    # intake embedding dim
    # intake encoder dim
    # intake input_word_count
    # intake target_word_count
    # intake training and test data

    # generate training model
    # train model

    # generate prediction models
    # return prediction models


    # organize run through:
    # TODO, it thinks input data should be 2D? Why?

    encoder_inputs = Input(shape=(None,))
    x = Embedding(in_toke_count, encoding_dim)(encoder_inputs)
    x, state_h, state_c = CuDNNLSTM(encoding_dim, return_state=True)(x)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,))
    
    x = Embedding(out_toke_count, encoding_dim)(decoder_inputs)
    decoder_lstm = CuDNNLSTM(encoding_dim, return_sequences=True)
    x = decoder_lstm(x, initial_state=encoder_states)

    decoder_dense = Dense(out_toke_count, activation='softmax')
    decoder_outputs=decoder_dense(x)

    # train weights:

    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    training_model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy')

    training_model.fit([enc_input, dec_input], target_set,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_split=val_split)

    # create prediction models
    self.encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(encoding_dim,))
    decoder_state_input_c = Input(shape=(encoding_dim,))
    decoder_input_states = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,
                                                     initial_state=decoder_input_states)

    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    self.decoder_model = Model([decoder_inputs] + decoder_input_states,
                               [decoder_outputs] + decoder_states)
    '''

    def one_hot(self, text):
        pass

    def predict(self, input_seq):
        # intake sequence
        # run sequence through encoder
        # create start of output sequence

        # run decoder with output sequence and encoder output

        # continue until max word length, or end tag is output

        # return output sequence

        state_values = self.encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1, len(self.output_index)))

        target_seq[0, 0, target_token_index['<start>']] = 1

        decoded_words = []

        while True:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + state_values)

            index = np.argmax(output_tokens[0, -1, :])
            predicted_word = self.output_reverse_index[index]
            decoded_words.append(predicted_word)

            if predicted_word == '<end>' or len(decoded_words) > self.max_pred_len:
                break

            # instead of resetting target_seq, add onto it?
            # resetting means it only gets 1 word of context
            
            # new_seq = np.zeros((1, 1, len(self.output_index)))
            # target_seq.append(new_seq)
            # target_seq[0, -1, index] = 1
            
            target_seq = np.zeros((1, 1, len(self.output_index)))
            target_seq[0, 0, index] = 1

            state_values = [h, c]

        return decoded_words




if __name__ == '__main__':
    # load data
    # generate indices
    # generate translator
    # train translator
    # test sequences

    data, input_uniques, target_uniques, inputs, targets = load_data('deu.txt', max_phrase_count=150)

    # vectorizing data
    # get variables

    input_forward, input_reverse = create_word_indices(input_uniques)
    target_forward, target_reverse = create_word_indices(target_uniques)

    max_input_len = max([len(txt.split(' ')) for txt in inputs])
    max_target_len = max([len(txt.split(' ')) for txt in targets])

    input_token_count = len(input_uniques)
    target_token_count = len(target_uniques)

    print(input_token_count, max_input_len, len(inputs))

    # generate vectors

    encoder_inputs = np.zeros((len(inputs), max_input_len, input_token_count), dtype='float32')
    decoder_inputs = np.zeros((len(targets), max_target_len, target_token_count), dtype='float32')
    decoder_targets = np.zeros((len(targets), max_target_len, target_token_count), dtype='float32')

    # fill vectors

    for i, (input_text, target_text) in enumerate(zip(inputs, targets)):
        for t, token in enumerate(input_text.split(' ')):
            encoder_inputs[i, t, input_forward[token]] = 1

        for t, token in enumerate(target_text.split(' ')):
            decoder_inputs[i, t, target_forward[token]] = 1
            if t > 1:
                decoder_targets[i, t-1, target_forward[token]] = 1

    print('decoder shape:')
    print(decoder_targets.shape)
    


    # create the model

    print(input_token_count, target_token_count)
    
    translator = Translator(input_forward, input_reverse, target_forward, target_reverse, 25)

    translator.create_models(256, 1024, encoder_inputs, decoder_inputs, decoder_targets,
                             64, 10, 0.2, input_token_count, target_token_count)
    # create_models(self, encoding_dim, embedding_dim, input_set, target_set, batch_size, epochs, val_split, in_toke_count, out_toke_count)







