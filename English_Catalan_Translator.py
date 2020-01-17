#!python3

'''

Toy english - catalan translator
    issue: only 638 sentence pairs, use different language?

'''

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


examples_count = 30000

BATCH_SIZE = 64

'''

preparing training data below

load data in from files
convert characters to ascii instead of unicode
create index-word encodings for both languages

create training dataset

'''

path_to_zip =  r"C:\Users\tonyt\Downloads\cat-eng.zip"
file_path = "cat.txt"

# convert unicode to ascii encoding
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


def create_dataset(path, num_examples):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    # why bother spliting by line above? why not just wait and split by tab later?
    # some sentences only split by line, some only split by tab?

    word_pairs = [[preprocess_sentence(s) for s in line.split('\t')]
                  for line in lines[:num_examples]]

    return word_pairs

class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab = set()

        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)

        self.word_to_id['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word_to_id[word] = index + 1

        for word, index in self.word_to_id.items():
            self.id_to_word[index] = word


def max_length(tensor):
    return max(len(t) for t in tensor)


def load_dataset(path, num_examples):
    # get cleaned input pairs
    pairs = create_dataset(path, num_examples)

    # index both languages using class defined above
    inp_lang = LanguageIndex(ca for en, ca in pairs)
    targ_lang = LanguageIndex(en for en, ca in pairs)

    # vectorize inputs and targets
    input_tensor = [[inp_lang.word_to_id[s] for s in ca.split(' ')]
                    for en, ca in pairs]

    target_tensor = [[targ_lang.word_to_id[s] for s in en.split(' ')]
                     for en, ca in pairs]

    max_length_inp = max_length(input_tensor)
    max_length_tar = max_length(target_tensor)

    # pad tensors out so they are all equal length
    
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen = max_length_inp,
                                                                 padding='post')

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen = max_length_tar,
                                                                  padding='post')

    

    return input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_tar
    

in_tensor, tar_tensor, input_lang, target_lang, max_input_length, max_target_length = load_dataset(file_path, examples_count)

input_tensor_x, input_tensor_y, target_tensor_x, target_tensor_y = train_test_split(in_tensor, tar_tensor, test_size = 0.3)

# print(len(input_tensor_x), len(input_tensor_y), len(target_tensor_x), len(target_tensor_y))


BUFFER_SIZE = len(input_tensor_x)
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_to_id)
vocab_tar_size = len(tar_lang.word_to_id)

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_x,
                                              target_tensor_x)).shuffle(BUFFER_SIZE)

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


def gru(units):
    # check to try and run on GPU
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_activation='sigmoid',
                                        recurrent_initializer='glorot_uniform')

    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, intial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units)
        self.fc = tf.keras.layers.Dense(vocab_size)

        # attention additions
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # encoder output shape == (batch_size, max_length, hidden_size)

        # hidden shape = (batch_size, max_length, 1)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)

        # layer below is added to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # attention weights shape after sum == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=1)

        output, state = self.gru(x)

        # output shape == batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)


optimizer = tf.train.AdamOptimizer()

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


EPOCHS = 10

for epoch in tange(EPOCHS):
    start = time.time()

    hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([targ_lang.word_to_id['<start>']] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                # pass encoder output to decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                dec_input = tf.expand_dims(targ[;,ArithmeticError t], 1)

            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss

            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            if batch % 100 == 0:
                print('Epoch: {}, Batch {}, Loss{:.4f}'.format(epoch + 1,
                                                               batch,
                                                               batch_loss.numpy()))

        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Epoch: {}, Loss {:.4f}'.format(epoch + 1,
                                              total_loss / N_BATCH))

        print('Time taken for 1 epoch: {} seconds\n'.format(time.time - start))


def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_to_id[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_to_id['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = deocer(dec_input, dec_hidden, enc_out)

        # storing attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.id_to_word[predicted_id] + ' '

        if targ_lang.id_to_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)


    return result, sentence, attention_plot


def plot_attention(attention, sentence, predicted_sentence):
    pass


def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_target):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

    print('Input: {}'.format(sentence))
    print('Predicted: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    plot_attention(attention_plot, sentence.split(' '), result.split(' '))





























