from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class TransformerModel(Model):
    def __init__(self, enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length,
                       h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super().__init__(**kwargs)

        # Set up the encoder
        self.encoder = Encoder(enc_vocab_size, enc_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)

        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v,
                               d_model, d_ff_inner, n, rate)

        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size)

    def padding_mask_self_attention(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        # return mask[:, newaxis, newaxis, :]
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        new_mask = maximum(mask, tf.transpose(mask, perm=(0,1,3,2)))
        return new_mask

    def padding_mask_cross_attention(self, input1, input2):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask1 = math.equal(input1, 0)
        mask1 = cast(mask1, float32)

        mask2 = math.equal(input2, 0)
        mask2 = cast(mask2, float32)

        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        # return mask[:, newaxis, newaxis, :]
        mask1 = mask1[:, tf.newaxis, :, tf.newaxis]
        mask2 = mask2[:, tf.newaxis, :, tf.newaxis]
        new_mask = maximum(mask2, tf.transpose(mask1, perm=(0,1,3,2)))
        return new_mask

    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, inputs, training=True):
        encoder_input, decoder_input = inputs

        # Create padding mask to mask the encoder inputs and the encoder
        # outputs in the decoder
        enc_padding_mask = self.padding_mask_self_attention(encoder_input)
        dec_padding_mask = self.padding_mask_cross_attention(encoder_input, decoder_input)

        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask_self_attention(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)

        # Feed the input into the encoder
        encoder_output = self.encoder(encoder_input, enc_padding_mask, training)

        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output,
                                      dec_in_lookahead_mask, dec_padding_mask, training)

        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)

        return model_output
