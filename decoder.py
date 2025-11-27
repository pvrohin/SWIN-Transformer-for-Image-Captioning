from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras import Model
from multi_head_attention import MultiHeadAttentionDec as MultiHeadAttention
from position_embedding import PositionEmbeddingFixedWeights
from encoder import AddNormalization, FeedForward
from tensorflow import cast, repeat, float32

from prefusion import Prefusion
 
# Implementing the Decoder Layer
class DecoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, name=None, **kwargs):
        super(DecoderLayer, self).__init__(name=name,**kwargs)
        self.multihead_attention1 = MultiHeadAttention(h, d_k, d_v, d_model,name=name+'mha1')
        self.dropout1 = Dropout(rate,name=name+'drop1')
        self.add_norm1 = AddNormalization(name=name+'normal1')
        self.multihead_attention2 = MultiHeadAttention(h, d_k, d_v, d_model,name=name+'mha2')
        self.dropout2 = Dropout(rate,name=name+'drop2')
        self.add_norm2 = AddNormalization(name=name+'normal2')
        self.feed_forward = FeedForward(d_ff, d_model,name=name+'ff1')
        self.dropout3 = Dropout(rate,name=name+'drop3')
        self.add_norm3 = AddNormalization(name=name+'normal3')
        self.prefuse = Prefusion(d_model,name=name+'prefusion')

    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))
 
    def call(self, x, encoder_output, lookahead_mask, global_feature, sequence_length, training):
        # Multi-head attention layer

        vg = cast(repeat(global_feature,sequence_length-1,axis=1),float32)
        x = self.prefuse(x,vg)

        # print("prefused shape: ", x.shape)

        multihead_output1 = self.multihead_attention1(x, x, x, lookahead_mask)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in a dropout layer
        multihead_output1 = self.dropout1(multihead_output1, training=training)
 
        # Followed by an Add & Norm layer
        addnorm_output1 = self.add_norm1(x, multihead_output1)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Followed by another multi-head attention layer
        multihead_output2 = self.multihead_attention2(addnorm_output1, encoder_output, encoder_output)
 
        # Add in another dropout layer
        multihead_output2 = self.dropout2(multihead_output2, training=training)
 
        # Followed by another Add & Norm layer
        addnorm_output2 = self.add_norm2(addnorm_output1, multihead_output2)
 
        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output2)
        # Expected output shape = (batch_size, sequence_length, d_model)
 
        # Add in another dropout layer
        feedforward_output = self.dropout3(feedforward_output, training=training)
 
        # Followed by another Add & Norm layer
        return self.add_norm3(addnorm_output2, feedforward_output)
 
# Implementing the Decoder
class Decoder(Model):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate,name=None, **kwargs):
        super(Decoder, self).__init__(name=name,**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model,name=name+'posembedding')
        self.dropout = Dropout(rate,name="decoder_dropout")
        self.decoder_layer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate,name='decoder_layer'+str(i)) for i in range(n)]
        self.sequence_length = sequence_length
 
    def call(self, output_target, encoder_output, lookahead_mask, global_feature, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(output_target)
        # Expected output shape = (number of sentences, sequence_length, d_model)
 
        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

 
        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.decoder_layer):
            x = layer(x, encoder_output, lookahead_mask, global_feature, self.sequence_length, training)
 
        return x


