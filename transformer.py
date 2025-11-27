from decoder import Decoder
from tensorflow import math, cast, float32, linalg, ones, maximum, newaxis
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from SWINblock import SwinTransformer
import tensorflow as tf
 
 
class TransformerModel(Model):
    def __init__(self,  dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate, name=None,**kwargs):
        super(TransformerModel, self).__init__(name=name,**kwargs)
 
        # Set up the encoder
        self.encoder = SwinTransformer(name=name+'SWINblock')
 
        # Set up the decoder
        self.decoder = Decoder(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff_inner, n, rate,name = name+'Decoder')
 
        # Define the final dense layer
        self.model_last_layer = Dense(dec_vocab_size,name=name+'Dense')

        self.dec_seq_length = dec_seq_length
 
    def padding_mask(self, input):
        # Create mask which marks the zero padding values in the input by a 1.0
        mask = math.equal(input, 0)
        mask = cast(mask, float32)
 
        # The shape of the mask should be broadcastable to the shape
        # of the attention weights that it will be masking later on
        return mask[:, newaxis, newaxis, :]
 
    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)
 
        return mask


    def build_graph(self,training):
        input_layer1 = tf.keras.Input(shape=(self.dec_seq_length-1))
        input_layer2 = tf.keras.Input(shape=(3,384,384))
        return Model(inputs=[input_layer2,input_layer1], outputs=self.call([input_layer2, input_layer1],training))

    def build(self, layer):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_initializer = TruncatedNormal(stddev=0.02,name=layer.name+'dense')
            #print(layer.name)
            if layer.bias is not None:
                layer.bias_initializer = Constant(0,name=layer.name+'bias')
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            layer.bias_initializer = Constant(0,name=layer.name+'beta')
            layer.gamma_initializer = Constant(1.0,name=layer.name+'gamma')
 
    def call(self, inputs, training):

        encoder_input, decoder_input = inputs
 
        # Create padding mask to mask the encoder inputs and the encoder outputs in the decoder

 
        # Create and combine padding and look-ahead masks to be fed into the decoder
        dec_in_padding_mask = self.padding_mask(decoder_input)
        dec_in_lookahead_mask = self.lookahead_mask(decoder_input.shape[1])
        dec_in_lookahead_mask = maximum(dec_in_padding_mask, dec_in_lookahead_mask)
 
        # Feed the input into the encoder
        encoder_output, encoder_output_global = self.encoder(encoder_input,training)
 


        # Feed the encoder output into the decoder
        decoder_output = self.decoder(decoder_input, encoder_output, dec_in_lookahead_mask, encoder_output_global, training)
 
        # Pass the decoder output through a final dense layer
        model_output = self.model_last_layer(decoder_output)
 
        return model_output
