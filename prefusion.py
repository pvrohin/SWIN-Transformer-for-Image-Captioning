from tensorflow import reshape, shape, matmul, math, transpose, cast, float32, concat, repeat
from tensorflow.keras.layers import Layer, Dense, ReLU, LayerNormalization
from keras.backend import softmax

class Prefusion(Layer):
    def __init__(self, d_model,name=None, **kwargs):
        super(Prefusion, self).__init__(name=name, **kwargs)

        self.activation = ReLU(name=name+'relu')
        self.Linear = Dense(d_model,name=name+'dense1')
        self.layer_norm = LayerNormalization(name=name+'layernorm')

 
    def call(self,x, vg):
        # Rearrange the queries to be able to compute all heads in parallel
        #check = self.W_q(queries)
        #print(f" projection shape: {check.shape}")
        x = self.layer_norm(x + self.activation(self.Linear( concat([x,vg], 2, name='concat'))))

        return x

#from numpy import random
#prefuse = Prefusion(512)

#x = random.random((64, 11,512))

#vg = random.random((64,1,512))

#vg = cast(repeat(vg,11,axis=1),float32)
#print(vg.shape)

#output = prefuse(x,vg)

#print(output.shape)