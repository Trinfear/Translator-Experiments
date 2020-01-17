#!python3


import re
import numpy as np
import unicodedata
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

'''

data prep for translators

    intake data
    split into phrases
    iterate through phrases[:max_phrase_count]
        clean each phrase, add start and end tags
        return list of sets of sentences (the sentence in each language)
    
    create language index for each language
    create vector representation for each phrase
    get max lengths for input and outputs
    add in padding

    return input_tensor, target_tensor, input_lang_index, out_lang_index,
           max_input_sentence_length, max_output_sentence_length

    get buffer_size, batch_count, input_token_count, output_token_count

    shuffle data by buffer_size
    run through train/test split    (where did test data get used??)
    run through tf.data.Dataset.from_tensor_slices  (shuffle again?)

    break dataset into batches and drop the remainder

'''


def unicode_to_ascii(sentence):
    return ''.join(c for c in unicodedata.normalize('NFD', sentence)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(sentence):
    sentence = unicode_to_ascii(sentence.lower().strip())

    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"^a-zA-Z?!,¿]+", " ", sentence)

    sentence = sentence.rstrip().strip()

    sentence = '<start> ' + sentence + ' <end>'

    return sentence


def trim_vocab(phrase_set, min_appearances=2, max_vocab=15000, min_vocab=10000):
    '''

    phrase_set = list of strings which are the sentences
    min_appearances = min number of times a word needs to appear not to be cut
    max_vocab = maximum number of words permited
    min_vocab = minimum word count for words to still get cut
    
    # iterate through phrases and add them all to a single list of words
    # get a dict of {unique_word: apearrance count}
    # get list of all words which appear less than min_appearances
        # iterate through phrases and remove any words on the list
    
    # if word count is still above max_vocab, call trim_vocab with min_appearances+1

    # remove just words, or whole phrases?

    '''

    # get list of all words and their appearance coutn
    vocab_list = []
    for phrase in phrase_set:
        sentence = phrase.split(' ')
        for word in sentence:
            vocab_list.append(word)

    # don't cut anything if word count is too low
    old_size = len(set(vocab_list))
    # print(old_size)
    # print(vocab_list[:15])
    
    if old_size < min_vocab:
        return phrase_set
    
    vocab_dict = Counter(vocab_list)
    del(vocab_list)

    # get list of invalid words
    invalid_words = []

    for word, count in vocab_dict.items():
        if count < min_appearances:
            invalid_words.append(word)

    # iterate through phrases and remove phrases which contain a banned word
    new_phrase_set = []

    for phrase in phrase_set:
        del_sentence = False
        sentence = phrase.split(' ')
        for word in sentence:
            if word in invalid_words:
                del_sentence = True
                continue
        if del_sentence == False:
            new_phrase_set.append(phrase)

    # check if word_count is below max_vocab

    vocab_set = set()
    for phrase in new_phrase_set:
        sentence = phrase.split(' ')
        for word in sentence:
            vocab_set.update(word)

    print(old_size, ' unique words cut down to ', len(vocab_set), ' unique words')
    print('minimum appearance count: ', min_appearances)
    
    if len(vocab_set) > max_vocab:
        new_phrase_set = trim_vocab(new_phrase_set,
                                    min_appearances = (min_appearances + 1))
        
    return new_phrase_set


class LanguageIndex():
    def __init__(self, phrase_set):
        self.phrase_set = phrase_set
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab = set()

        self.create_indices()

    def create_indices(self):
        for phrase in self.phrase_set:
            self.vocab.update(phrase.split(' '))

        self.vocab = sorted(self.vocab)
        self.word_to_id['<pad>'] = 0
        self.id_to_word[0] = '<pad>'

        for index, word in enumerate(self.vocab):
            self.word_to_id[word] = index + 1
            self.id_to_word[index + 1] = word


def get_dataset(path, example_count=1000000):
    # TODO: add in returning a set of all unique words in a language
    data = open(path, encoding='UTF-8').read().strip().split('\n')
    data = [[preprocess_sentence(sentence) for sentence in datum.split('\t')]
            for datum in data[:example_count]]

    # TODO: iterate through phrases and get count for each word
    # remove words with low comaniltiy from phrases entirely
    lang_a = [a for a, b in data]
    lang_b = [b for a, b in data]

    lang_a = trim_vocab(lang_a)
    lang_b = trim_vocab(lang_b)

    input_lang_index = LanguageIndex(lang_a)
    output_lang_index = LanguageIndex(lang_b)

    input_uniques = list(input_lang_index.vocab)
    output_uniques = list(output_lang_index.vocab)

    # onehot just not necessary with embedding layer, problem solved
    input_tensor = [[input_lang_index.word_to_id[w] for w in sent.split(' ')] for sent in lang_a]

    target_tensor = [[output_lang_index.word_to_id[w] for w in sent.split(' ')] for sent in lang_b]

    max_input_length = max(len(t) for t in input_tensor)
    max_output_length = max(len(t) for t in target_tensor)

    input_tensor = pad_sequences(input_tensor, maxlen=max_input_length,
                                 padding='post')

    target_tensor = pad_sequences(input_tensor, maxlen=max_output_length,
                                  padding='post')

    return input_tensor, target_tensor, input_lang_index, output_lang_index, max_input_length, max_output_length, input_uniques, output_uniques



def get_reverse_dataset():
    # basically get dataset, but returns opposite input/output pairs?
    # instead can you just basically relabel the other ones?
        # that's probably better, but will leave this function if something unexpected
    pass



if __name__ == '__main__':
    pass
