#!python3

'''

TODO: improve comments explaining what type and shape data is as it progresses


experimenting with do it yourself seq 2 seq models

intake data
create onehot lexicon for words
convert sentences to arrays of onehot
feed to encoder
feed encoder output to decoder
set target value as onehot arrays in target language

'''

import re
import numpy as np
import unicodedata


# import and clean data


class LanguageLexicon():
    # intakes a word set for a language
    # creates onehot encoder and decoder for word set
    # because numpy arrays can't be keys uses index of onehot as key for id->word
    def __init__(self, word_set, language='undefined'):
        self.language = language  # just a string tagging what language is used
        self.word_set = word_set
        self.word_to_id, self.id_to_word = self.create_one_hot(word_set)

    def create_one_hot(self, word_set):
        word_to_id = {}
        id_to_word = {}

        size = len(word_set)
        
        word_to_id['<pad>'] = np.zeros(size, dtype=np.int8)
        id_to_word[0] = '<pad>'

        word_set.remove('<pad>')

        for i in range(size):
            if i == 0:
                continue
            
            word = word_set[i]
            vector = np.zeros(size, dtype=np.int8)
            vector[i] = 1
            
            word_to_id[word] = vector
            id_to_word[i] = word

        return word_to_id, id_to_word


def organize_data(data):
    # break up based by line
    # break into languages by tab
    # returns list of phrases
    # each phrase is a list of strings, one for each language
    data = data.split('\n')

    data = [phrase.split('\t') for phrase in data]

    return data

def unicode_to_ascii(sentence):
    return ''.join(c for c in unicodedata.normalize('NFD', sentence)
                   if unicodedata.category(c) != 'Mn')


def clean_data(data):
    # removes special characters
    # adds spaces between punctuation and tags so they are recognized as unique words
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

            cleaned_phrases.append(phrase)
            
        cleaned_data.append(cleaned_phrases)

    return cleaned_data


def create_language_lexicons(data):
    # intake data in form list of phrases(list of two cleaned strings)
    # get a list of all unique words
    # pass to language_lexicon
    language_one = []
    language_two = []
    
    for phrase_set in data:
        for word in phrase_set[0].split(' '):
            language_one.append(word)

        for word in phrase_set[1].split(' '):
            language_two.append(word)

    # does not add <pad>!!!!!

    language_one = list(set(language_one))
    lexicon_one = LanguageLexicon(language_one)

    language_two = list(set(language_two))
    lexicon_two = LanguageLexicon(language_two)

    return lexicon_one, lexicon_two


def create_vector_data(data, lexicon_one, lexicon_two):
    # intake data
    # iterate through phrases and convert them to vector sets
    # returns list of phrases(list of two numpy 2d arrays)
    vector_sets = []
    for phrase_set in data:
        vector_set_one = []
        vector_set_two = []
        
        for word in phrase_set[0].split(' '):
            vector_set_one.append(lexicon_one.word_to_id[word])
        
        vector_set_one = np.array(vector_set_one)

        for word in phrase_set[1].split(' '):
            vector_set_two.append(lexicon_two.word_to_id[word])

        vector_set_two = np.array(vector_set_two)

        vector_sets.append([vector_set_one, vector_set_two])

    return vector_sets


def get_data(data_dir, max_phrase_count=None):
    # load data
    # organize data
    # clean data
    # create lexicons
    # create training sets
    # return lexicons and data sets
    # data sets are 
    data = open(data_dir, encoding='UTF-8').read().strip()
    data = organize_data(data)

    if max_phrase_count:
        data = data[:max_phrase_count]

    data = clean_data(data)

    lexicon_one, lexicon_two = create_language_lexicons(data)

    vector_sets = create_vector_data(data, lexicon_one, lexicon_two)

    # should shuffle vector sets

    return lexicon_one, lexicon_two, vector_sets, data


if __name__ == '__main__':

    data = open('deu.txt', encoding='UTF-8').read().strip()
    data = organize_data(data)

    data = data[:15]
    
##    for datum in data:
##        print(datum)
##
##    print('\n\n\n')
        
    data = clean_data(data)

    for datum in data:
        print(datum)

    print('\n\n\n')

    lexicon_one, lexicon_two = create_language_lexicons(data)

##    print(lexicon_one.word_set)
##    print('\n\n\n')
##    print(lexicon_two.word_set)
##    for key, item in lexicon_two.word_to_id.items():
##        print(key)

    data_set = create_vector_data(data, lexicon_one, lexicon_two)

    print(data_set[0])




