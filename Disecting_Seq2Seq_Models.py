
import tensorflow as tf

from keras.layers import Dense, GRU, Input, CuDNNLSTM


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()
        
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        self.gru = GRU(dec_units, 
                       return_sequences=True, 
                       return_state=True, 
                       recurrent_activation='sigmoid', 
                       recurrent_initializer='glorot_uniform')
        
        self.output_layer = Dense(vocab_size)

        # attention weights
        self.W1 = Dense(self.dec_units)
        self.W2 = Dense(self.dec_units)
        self.verdict = Dense(1)

    def call(self, x, hidden, enc_output):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = self.W1(enc_output) + self.W2(hidden_with_time_axis)
        score = self.verdict(tf.nn.tanh(score))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights











class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder,  self).__init__()
        self.batch_size = batch_sz
        self.encoder_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = GRU(dec_units, 
                       return_sequences=True, 
                       return_state=True, 
                       recurrent_activation='sigmoid', 
                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, intial_state = hidden)
        return output, state

    def initialize_hidden_State(self):
        return tf.zeros((self.batch_size, self.encoder_units))





















encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = CuDNNLSTM(encoding_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None, dum_decoder_tokens))

decoder_lstm = CuDNNLSTM(encoder_dim, return_sequences=True, return_state=True)
decoder_outputs, d_state_h, d_state_c = decoder_lstm(decoder_inputs,
                                                     intial_state=encoder_states)

decoder_dense = Dense(num_encoder_toekns, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)





encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
x, state_h, state_c = LSTM(encoder_dims,
                           return_state=True)(x)

encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None,))
x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x = CuDNNLSTM(encoder_dims, return_sequences=True)(x, intial_state=encoder_states)
decoder_outputs = Dense(num_decoder_tokens, activation='Softmax')(x)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)













