#!python3

'''

seq 2 seq translator following the keras tutorial instead of tensorflow

initially works character to character but should be upgraded word to word

'''



from __future__ import print_function

from keras.models import Model
from keras.layers import Input, CuDNNLSTM, LSTM, Dense
import numpy as np


# Global Variables

batch_size = 64
epochs = 10
encoding_dim = 256
num_samples = 30000

data_path = 'deu.txt'


# preparing data
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    
    target_text = '\t' + target_text + '\n'
    
    input_texts.append(input_text)
    target_texts.append(target_text)

    # break into words instead of characters

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])

target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])


encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1


# create model
# add embedding layers for words
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# add an embedding layer here
encoder = LSTM(encoding_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None, num_decoder_tokens))

# add an embedding layer here
decoder_lstm = LSTM(encoding_dim, return_sequences=True, return_state=True)
decoder_outputs, d_state_h, d_state_c = decoder_lstm(decoder_inputs,
                                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
print(decoder_outputs.shape)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# run training
# might need to update keras
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# Save Model
model.save('s2s_1.h5')


# running on a new sentence
# 1) encode input and retrieve intial decoder state
# 2) run one step of decoder with this initial state, and with a 'start of
# sequence' token as target
# 3) repeat with the current target token and current states


# define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(encoding_dim,))
decoder_state_input_c = Input(shape=(encoding_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


# reverse-lookup token index to decode sequences back to something readable
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())

reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # encode input
    states_value = encoder_model.predict(input_seq)

    # generate empty target sequence
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # set the first input as the sentence start token
    target_seq[0, 0, target_token_index['\t']] = 1

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
            len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1

        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    input_seq = encoder_input_data[seq_index: seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)














