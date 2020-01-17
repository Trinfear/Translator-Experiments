#!python3


'''

get datasets for several langauge combos

use models in simple_TF_V2 for encoders and decoders

for each language dataset create translation sets for both directions
    ie english --> french and french --> english

create a single encoder and a single decoder for every language
    create a class which contains language indices and enc/dec for each lang?

train all together, such that they all learn a single thought vector

'''

import tensorflow as tf
import Simple_TF_V2 as Trans
import Translator_Data_Prep_V2 as dataprep
from sklearn.model_selection import train_test_split



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# define file paths =====================================================================

german_path = 'deu.txt'
catalan_path = 'cat.txt'



# define global variables ===============================================================

batch_size = 4
embedding_dim = 512
units = 256
train_epochs = 10

max_samples = 24000

# load and prep data ====================================================================
    # stick all this in a func which intakes *argv paths
    # then iterate through paths and prepare all this for each one?

def concat_language(*argv):
    # intake all uniques for language from every source
    # iterate through uniques and add them to a single set
    # get length of set
    all_vocab = set()

    for arg in argv:
        all_vocab.update(arg)

    all_vocab = list(all_vocab)

    new_lexicon = dataprep.LanguageIndex(all_vocab)

    return len(all_vocab), all_vocab, new_lexicon

'''
# load data
(german_english_vector, german_german_vector, german_english_index,
 german_german_index, german_max_english, german_max_german,
 german_english_uniques, german_german_uniques) = dataprep.get_dataset(german_path)

(catalan_english_vector, catalan_catalan_vector, catalan_english_index,
 catalan_catalan_index, catalan_max_english, catalan_max_catalan,
 catalan_english_uniques, catalan_catalan_uniques) = dataprep.get_dataset(catalan_path)

# get total vocab size for each language
english_vocab_size, english_uniques = get_vocab_size(german_english_uniques, catalan_english_uniques)
german_vocab_size, german_uniques = get_vocab_size(german_german_uniques)
catalan_vocab_size, catalan_uniques = get_vocab_size(catalan_catalan_uniques)


# create final dataset for each language
(german_english_train, german_english_test, german_german_train,
 german_german_test) = train_test_split(german_english_vector,
                                        german_german_vector, test_size = 0.3)

german_buffer_size = len(german_english_train)
english_to_german_dataset = tf.data.Dataset.from_tensor_slices((german_english_train,
                                                                german_german_train)).shuffle(german_buffer_size)

german_to_english_dataset = tf.data.Dataset.from_tensor_slices((german_german_train,
                                                                german_english_train)).shuffle(german_buffer_size)


(catalan_english_train, catalan_english_test, catalan_catalan_train,
 catalan_catalan_test) = train_test_split(catalan_english_vector,
                                          catalan_catalan_vector, test_size = 0.3)

catalan_buffer_size = len(catalan_english_train)

english_to_catalan_dataset = tf.data.Dataset.from_tensor_slices((catalan_english_train,
                                                                 catalan_catalan_train)).shuffle(catalan_buffer_size)

'''

# func here is to replace this whole section with more reusable code
def load_and_prep_data(*argv):
    # intakes paths to datafiles as arguments
    # iterate through each path
        # load data and prep it
        # split into train test
        # generate datasets for forwards and backwards translations
    # return all datasets
    all_data_sets = []
    all_indices = []
    extra_variables = []
    all_batch_counts = []

    # label by languages inside by assuming input is english and target is arg - '.txt'?

    for arg in argv:
        (inp_tensor, targ_tensor, inp_index, targ_index, inp_max, targ_max,
         inp_uniques, targ_uniques) =  dataprep.get_dataset(arg, example_count=max_samples)
        
        '''

        inp_tensor = vector form of all input sentences
        targ_tensor = vector form of all target sentences
        inp_index = language index for input language
        targ_index = language index for target language
        inp_max = maximum length of any input tensor
        inp_uniques = all unique vocab in input language
        targ_uniques = all unique vocab in target language

        '''

        inp_train, inp_test, targ_train, targ_test = train_test_split(inp_tensor,
                                                                      targ_tensor,
                                                                      test_size = 0.3)

        buffer_size = len(inp_train)
        batch_count_one = buffer_size//batch_size

        inp_to_targ_data = tf.data.Dataset.from_tensor_slices((inp_train,
                                                               targ_train)).shuffle(buffer_size)

        buffer_size = len(targ_train)
        batch_count_two = buffer_size//batch_size
        
        targ_to_inp_data = tf.data.Dataset.from_tensor_slices((targ_train,
                                                               inp_train)).shuffle(buffer_size)

        inp_to_targ_data = inp_to_targ_data.batch(batch_size, drop_remainder=True)
        targ_to_inp_data = targ_to_inp_data.batch(batch_size, drop_remainder=True)

        all_data_sets.append((inp_to_targ_data, targ_to_inp_data))
        all_indices.append((inp_index, targ_index))
        extra_variables.append((inp_max, targ_max, inp_uniques, targ_uniques))
        all_batch_counts.append((batch_count_one, batch_count_two))

    # returns lists of sets of datasets, list of language indices, and extra variables
        # each set is (inp_to_targ, targ_to_inp) vector
        # each indices set is (inp_index, targ_inex)
        # extra variables is a set of (inp_max, targ_max, inp_uniques, targ_uniques)
        # batch counts is the number of batches in each corrsponding dataset
    
    return all_data_sets, all_indices, extra_variables, all_batch_counts


datasets, indices_set, extra, batches = load_and_prep_data(german_path,
                                                           catalan_path)

# labeling languages has to be done outside?
english_to_german_data = datasets[0][0]
german_to_english_data = datasets[0][1]

english_to_catalan_data = datasets[1][0]
catalan_to_english_data = datasets[1][1]

english_lexicon_german = indices_set[0][0]
german_lexicon = indices_set[0][1]

english_lexicon_catalan = indices_set[1][0]
catalan_lexicon = indices_set[1][1]

# get total vocab size for each language
    # need to make new unified index for english?

english_german_uniques = english_lexicon_german.vocab
english_catalan_uniques = english_lexicon_catalan.vocab

print(len(english_german_uniques), len(english_catalan_uniques))

german_vocab = german_lexicon.vocab

catalan_vocab = catalan_lexicon.vocab


english_word_count, full_english_vocab, full_english_lexicon = concat_language(english_german_uniques, english_catalan_uniques)
german_word_count = len(german_vocab)
catalan_word_count = len(catalan_vocab)

print(english_word_count)
print('\n\n\n')



# create models =========================================================================
    # get total vocab sizes across all pairings a language appears in
    # create an encoder and decoder for each language

english_encoder = Trans.Encoder(english_word_count, embedding_dim, units, batch_size)
english_decoder = Trans.Decoder(english_word_count, embedding_dim, units, batch_size)

german_encoder = Trans.Encoder(german_word_count, embedding_dim, units, batch_size)
german_decoder = Trans.Decoder(german_word_count, embedding_dim, units, batch_size)

catalan_encoder = Trans.Encoder(catalan_word_count, embedding_dim, units, batch_size)
catalan_decoder = Trans.Decoder(catalan_word_count, embedding_dim, units, batch_size)

# func below is just a generalization of code above
    # not currently used, but would be useful with more languages
def create_models(*argv):
    models = []
    for word_count in argv:
        encoder = Trans.Encoder(word_count, embedding_dim, units, batch_size)
        decoder = Trans.Decoder(word_count, embedding_dim, units, batch_size)
        
        models.append((encoder, decoder))
    
    return models


# train models ==========================================================================
    # train an encoder and decoder both ways for each language pairing


def train_models(models, data_sets, indices, batch_counts, epochs,
                 optimizer, batch_size):
    # NOTE: each direction requires a seperate pair, ie english to german is one set,
        # and running german to english requires another set of inputs
    
    # models is a list of encoder decoder pairs (encoder, decoder)
    # data_sets is a corresponding list of datasets
    # indices is a corresponding list of index pairs (inp_index, targ_index)
    # batch counts is a corresponding list of batch counts
    # epochs is epoch count for training
    # batch_size is the size of each batch
    print(batch_counts)
    
    assert len(models) == len(data_sets) == len(indices) == len(batch_counts)

    trained_models = []
    print('made it here')
    print(len(models))

    for i in range(len(models)):
        encoder = models[i][0]
        decoder = models[i][1]

        data = data_sets[i]

        inp_index = indices[i][0]
        targ_index = indices[i][1]

        N_batches = batch_counts[i]

        encoder, decoder = Trans.train_networks(encoder, decoder, epochs, data,
                                                optimizer, inp_index, targ_index,
                                                batch_size, N_batches)

        trained_models.append((encoder, decoder))

        encoder.save('enc_{}'.format(i))
        decoder.save('dec_{}'.format(i))

        if i < (len(models) - 1):
            print('\n\nnext model set: \n')

    return trained_models


model_set = [(english_encoder, german_decoder),
             (german_encoder, english_decoder),
             (english_encoder, catalan_decoder),
             (catalan_encoder, english_decoder)]

data_set = [english_to_german_data, german_to_english_data,
            english_to_catalan_data, catalan_to_english_data]

index_set = [(full_english_lexicon, german_lexicon),
             (german_lexicon, full_english_lexicon),
             (full_english_lexicon, catalan_lexicon),
             (catalan_lexicon, full_english_lexicon)]

batch_count_set = [batches[0][0],
                   batches[0][1],
                   batches[1][0],
                   batches[1][1]]

optimizer = tf.train.AdamOptimizer()

train_models(model_set, data_set, index_set, batch_count_set, train_epochs, optimizer,
             batch_size)


encoder_dict = {'english': english_encoder,
                'german': german_encoder,
                'catalan': catalan_encoder}

decoder_dict = {'english': english_decoder,
                'german': german_decoder,
                'catalan': catalan_decoder}

index_dict = {'english': full_english_lexicon,
              'german': german_lexicon,
              'catalan': catalan_lexicon}


def translate(enc_lang, dec_lang, sent):

    enc_lang = enc_lang.lower()
    dec_lang = dec_lang.lower()
    
    encoder = encoder_dict[enc_lang]
    decoder = decoder_dict[dec_lang]

    inp_index = index_dict[enc_lang]
    dec_index = index_dict[dec_lang]

    Trans.translate(sent, encoder, decoder, inp_index, dec_index, units)


# test models ===========================================================================
    # test several sentences from each language to each other language

















