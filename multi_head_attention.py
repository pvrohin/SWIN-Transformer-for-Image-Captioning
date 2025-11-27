from tensorflow import reshape, shape, matmul, math, transpose, cast, float32
from tensorflow.keras.layers import Layer, Dense
from keras.backend import softmax
 
# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self,name=None, **kwargs):
        super(DotProductAttention, self).__init__(name=name,**kwargs)
 
    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
 
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
 
        # Computing the weights by a softmax operation
        weights = softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)


# Implementing the Multi-Head Attention
class MultiHeadAttentionDec(Layer):
    def __init__(self, h, d_k, d_v, d_model,name=None, **kwargs):
        super(MultiHeadAttentionDec, self).__init__(name=name,**kwargs)
        self.attention = DotProductAttention(name=name+'dotproductattention')  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Dense(d_k,name=name+'Wq')  # Learned projection matrix for the queries
        self.W_k = Dense(d_k,name=name+'Wk')  # Learned projection matrix for the keys
        self.W_v = Dense(d_v,name=name+'Wv')  # Learned projection matrix for the values
        self.W_o = Dense(d_model,name=name+'Wo')  # Learned projection matrix for the multi-head output
 
    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x
 
    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        #check = self.W_q(queries)
        #print(f" projection shape: {check.shape}")
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # print("q shape: ", q_reshaped.shape)
        #print("mask: ", mask.shape)
        #print(f" reshaped shape: {q_reshaped.shape}")
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        #print(f"Attention output shape: {output.shape}")
        return self.W_o(output)

