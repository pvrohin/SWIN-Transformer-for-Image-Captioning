import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras.utils import plot_model


def to_2tuple(x):
    return (x, x)

def DropPath(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (tf.size(x) - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = tf.cast(tfp.distributions.Bernoulli(probs=keep_prob).sample(sample_shape=x.shape),tf.float32)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = tf.cast(random_tensor/keep_prob,tf.float32)

    return x * random_tensor


def window_partition(x, window_size=6):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size (Default: 7)
        
    Returns:
        windows: (num_windows * B, window_size, window_size, C)
                 (8*8*B, 7, 7, C)
    """
    
    B, H, W, C = x.shape
    B =tf.shape(x)[0]
    #print("B: ",B)
    # Convert to (B, 8, 7, 8, 7, C) 
    # x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    
    # Convert to (B, 8, 8, 7, 7, C)
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    
    # Efficient Batch Computation - Convert to (B*8*8, 7, 7, C)
    # windows = windows.view(-1, window_size, window_size, C)
    windows = tf.reshape(windows, [-1, window_size, window_size, C])
    
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows * B, window_size, window_size, C)
                 (8*8*B, 7, 7, C)
        window_size (int): window size (default: 7)
        H (int): Height of image (patch-wise)
        W (int): Width of image (patch-wise)
        
    Returns:
        x: (B, H, W, C)
    """
    
    # Get B from 8*8*B
    #print((windows.shape[0]))
    if windows.shape[0] is not None:
        B = int(windows.shape[0]/ (H * W / window_size / window_size))
    else :
        B = tf.shape(windows)[0]
    
    # Convert to (B, 8, 8, 7, 7, C)
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])

    # Convert to (B, 8, 7, 8, 7, C)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    
    # Convert to (B, H, W, C)
    # x = x.view(B, H, W, -1)
    x = tf.reshape(x,[B, H, W, -1])
    
    return x

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=tf.keras.activations.gelu, drop=0.,name=None):
        super(Mlp,self).__init__(name=name)
        #self.name= name
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        denseone_name = name+'dense1'
        self.fc1 = tf.keras.layers.Dense(hidden_features, activation=tf.keras.activations.gelu,name=denseone_name)
        # self.act_layer = act_layer()
        dense2_name = name+'dense2'
        self.fc2 = tf.keras.layers.Dense(out_features, activation=None,name=dense2_name)
        drop_name = name+'drop'
        self.drop = tf.keras.layers.Dropout(drop,name=drop_name)
        
    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(tf.keras.layers.Layer):
    """ Window based multi-head self attention(W-MSA) module with relative position bias.
        Used as Shifted-Window Multi-head self-attention(SW-MSA) by providing shift_size parameter in
        SwinTransformerBlock module
        
    Args:
        dim (int): Number of input channels (C)
        window_size (tuple[int]): The height and width of the window (M)
        num_heads (int): Number of attention heads for multi-head attention
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v (Default: True)
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight (Default: 0.0)
        proj_drop (float, optional): Dropout ratio of output (Default: 0.0)
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,name=None):
        super(WindowAttention,self).__init__(name=name)
        self.dim = dim
        self.window_size = window_size # Wh(M), Ww(M) (7, 7)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Parameter table of relative position bias: B_hat from the paper
        # (2M-1, 2M-1, num_heads) or (2*Wh-1 * 2*W-1, num_heads)
        self.relative_position_bias_table = tf.Variable(tf.zeros([(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads]), trainable=True)
        
        # Pair-wise relative position index for each token inside the window
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w)) # (2, M, M) or (2, Wh, Ww)
        coords_flatten = tf.reshape(coords, [2, window_size[0]*window_size[1]]) # (2, M^2)
        
        # None is dummy dimension
        # coords_flatten[:, :, None] = (2, M^2, 1)
        # coords_flatten[:, None, :] = (2, 1, M^2)
        # relative_coords = (2, M^2, M^2)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        
        # (2, M^2, M^2) -> (M^2, M^2, 2)
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        temp = relative_coords.numpy()
        temp[:, :, 0] += self.window_size[0] - 1 # make it start from 0 index
        temp[:, :, 1] += self.window_size[1] - 1
        temp[:, :, 0] *= 2 * self.window_size[1] - 1 # w.r.t x-axis
        relative_coords = tf.convert_to_tensor(temp, dtype=np.int32)
        
        # x-axis + y-axis
        relative_position_index = tf.reduce_sum(relative_coords, -1)
        
        self.relative_position_index = relative_position_index
        
        # Attention
        querykeyvalue_name = name + 'querykeyvalue_layer'

        self.qkv = tf.keras.layers.Dense(dim*3, use_bias=qkv_bias,name = querykeyvalue_name) # W_Q, W_K, W_V
        #dropout_name = name+'attn_drop'
        #self.attn_drop = tf.keras.layers.Dropout(attn_drop,name=dropout_name)
        dense_name = name+ 'dense_layer'
        self.proj = tf.keras.layers.Dense(dim,name=dense_name)
    
        dropout2_name = name + 'drop2'
        self.proj_drop = tf.keras.layers.Dropout(proj_drop,name=dropout2_name)
        
        truncname = name+'trunc'
        tf.random.truncated_normal(self.relative_position_bias_table.shape, stddev=.02,name=truncname)
        # self.softmax = tf.keras.activations.softmax(axis=-1)
        
    
    def call(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), N refers to number of patches in a window (M^2)
            mask: (0/-inf) mask with shape of (num_windows, M^2, M^2) or None
                  -> 0 means applying attention, -inf means removing attention
        """
        # (batch, M^2, C)
        B_, N, C = x.shape
        B_ = tf.shape(x)[0]
        # (num_windows*B, N, 3C)
        qkv = self.qkv(x)
    
        # (B, N, 3, num_heads, C // num_heads)
        # qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)

        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])

        # Permute to (3, B_, num_heads, N, C // num_heads)
        '''
        3: referring to q, k, v (total 3)
        B: batch size
        num_heads: multi-headed attention
        N:  M^2, referring to each token(patch)
        C // num_heads: Each head of each of (q,k,v) handles C // num_heads -> match exact dimension for multi-headed attention
        '''
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
        # Decompose to query/key/vector for attention
        # each of q, k, v has dimension of (B_, num_heads, N, C // num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2] # Why not tuple-unpacking?
        
        q = q * self.scale
        
        # attn becomes (B_, num_heads, N, N) shape
        # N = M^2

        k = tf.transpose(k, perm=[0, 1, 3, 2])
        attn = (q @ k)
        
        # Remember that relative_position_bias_table = ((2M-1)*(2M-1), num_heads), B_hat from the paper
        # relative_position_index's elements are in range [0, 2M-2]
        # Convert to (M^2, M^2, num_heads). This is B matrix from the paper
        #print("Relative position index", self.relative_position_index.shape)
        a = tf.reshape(self.relative_position_index, [-1])
        #print("a: ",a.shape)
        b = tf.gather(self.relative_position_bias_table, a)
        a = tf.reshape(b, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = a
        # Convert to (num_heads, M^2, M^2) to match the dimension for addition
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        
        # (B, num_heads, N, N) + (1, num_heads, M^2, M^2), where N=M^2
        # attn becomes (B_, num_heads, N, N) or (B, num_heads, M^2, M^2)

        attn = attn + tf.expand_dims(relative_position_bias, 0)
        
        if mask is not None:
            nW = mask.shape[0] # nW = number of windows
            
            # attn.view(...) = (B, nW, num_heads, N, N)
            # mask.unsqueeze(1).unsqueeze(0) = (1, num_windows, 1, M^2, M^2)
            # So masking is broadcasted along B and num_heads axis which makes sense
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N]) + tf.expand_dims(tf.expand_dims(mask, 1), 0)
            
            # attn = (nW * B, num_heads, N, N)
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
            attn = tf.keras.activations.softmax(attn)
        else:

            attn = tf.keras.activations.softmax(attn)
            
        attn = tf.keras.activations.softmax(attn)
        
        # attn = (nW*B, num_heads, N, N)
        # v = (B_, num_heads, N, C // num_heads). B_ = nW*B
        # attn @ v = (nW*B, num_heads, N, C // num_heads)
        # (attn @ v).transpose(1, 2) = (nW*B, N, num_heads, C // num_heads)
        # Finally, x = (nW*B, N, C), reshape(B_, N, C) performs concatenation of multi-headed attentions
        x = (attn @ v)
        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), [B_, N, C])
        
        # Projection Matrix (W_0). dim doesn't change since we used C // num_heads for MSA
        # x = (B_, N, C)
        x = self.proj(x)
        #check = np.asarray(self.proj.weights)
        #print("weights: ", self.proj.weights[0][0][0]) ##len(self.proj.weights), len(self.proj.weights[0]), len(self.proj.weights[0][0]))
        x = self.proj_drop(x)
        #print("XXXXXXX", x[0][0][0])
        return x        


class WindowAttention2(tf.keras.layers.Layer):
    """ Window based multi-head self attention(W-MSA) module with relative position bias.
        Used as Shifted-Window Multi-head self-attention(SW-MSA) by providing shift_size parameter in
        SwinTransformerBlock module
        
    Args:
        dim (int): Number of input channels (C)
        window_size (tuple[int]): The height and width of the window (M)
        num_heads (int): Number of attention heads for multi-head attention
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v (Default: True)
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight (Default: 0.0)
        proj_drop (float, optional): Dropout ratio of output (Default: 0.0)
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., name=None):
        super(WindowAttention2,self).__init__(name=name)
        self.dim = dim
        self.window_size = window_size # Wh(M), Ww(M) (7, 7)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # Parameter table of relative position bias: B_hat from the paper
        # (2M-1, 2M-1, num_heads) or (2*Wh-1 * 2*W-1, num_heads)
        self.relative_position_bias_table = tf.Variable(tf.zeros([(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads]), trainable=True)
        
        # Pair-wise relative position index for each token inside the window
        coords_h = tf.range(self.window_size[0])
        coords_w = tf.range(self.window_size[1])
        coords = tf.stack(tf.meshgrid(coords_h, coords_w)) # (2, M, M) or (2, Wh, Ww)

        coords_flatten = tf.reshape(coords, [2, window_size[0]*window_size[1]]) # (2, M^2)
        global_coord = tf.constant([[0],[0]])


        coords_flatten1 = tf.concat([coords_flatten,global_coord],axis=1)

        # None is dummy dimension
        # coords_flatten[:, :, None] = (2, M^2, 1)
        # coords_flatten[:, None, :] = (2, 1, M^2)
        # relative_coords = (2, M^2, M^2)
        relative_coords = coords_flatten[:, :, None] - coords_flatten1[:, None, :]
        
        # (2, M^2, M^2) -> (M^2, M^2, 2)
        relative_coords = tf.transpose(relative_coords, perm=[1, 2, 0])
        temp = relative_coords.numpy()
        temp[:, :, 0] += self.window_size[0] - 1 # make it start from 0 index
        temp[:, :, 1] += self.window_size[1] - 1
        temp[:, :, 0] *= 2 * self.window_size[1] - 1 # w.r.t x-axis
        relative_coords = tf.convert_to_tensor(temp, dtype=np.int32)
        # x-axis + y-axis
        relative_position_index = tf.reduce_sum(relative_coords, -1)


        
        self.relative_position_index = relative_position_index
        
        keyvalue_name = name + 'keyvalue_layer'
        queryname = name + 'querylayer'
        # Attention
        self.kv = tf.keras.layers.Dense(dim*2, use_bias=qkv_bias, name=keyvalue_name) # W_Q, W_K, W_V
        self.q = tf.keras.layers.Dense(dim, use_bias=qkv_bias,name=queryname)

        #dropoutname = name + 'dropout' 
        #self.attn_drop = tf.keras.layers.Dropout(attn_drop,name=dropoutname)

        #projname = name + 'Dense_proj'
        #self.proj = tf.keras.layers.Dense(dim,name=projname)
        #proj_dropname = name +'proj_dropname'
        #self.proj_drop = tf.keras.layers.Dropout(proj_drop,name=proj_dropname)

        truncname = name+ 'truncvalues'
        
        tf.random.truncated_normal(self.relative_position_bias_table.shape, stddev=.02,name = truncname )
        # self.softmax = tf.keras.activations.softmax(axis=-1)
        
    
    def call(self, x, xtotal, xavg, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C), N refers to number of patches in a window (M^2)
            mask: (0/-inf) mask with shape of (num_windows, M^2, M^2) or None
                  -> 0 means applying attention, -inf means removing attention
        """
        # (batch, M^2, C)

        # (num_windows*B, N, 3C)


        B_ = tf.shape(x)[0]
        xavgR = tf.repeat(xavg,repeats=B_//tf.shape(xavg)[0], axis=0)

        x_c = tf.concat([x,xavgR],axis=1)

        kv = self.kv(x_c)
        q = self.q(x)
        q_avg = self.q(xavg)
        kvglobal = self.kv(xtotal)

        B, Ntotal, C = xtotal.shape
        B = tf.shape(xtotal)[0]

        B_, N, C = x_c.shape
        B_ = tf.shape(x_c)[0]
        BG_, NG, CG = x.shape
        BG_ = tf.shape(x)[0]
        
        

        # (B, N, 3, num_heads, C // num_heads)
        # qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)


        kv = tf.reshape(kv, [B_, N, 2, self.num_heads, C // self.num_heads])
        q = tf.reshape(q,[BG_, NG, 1, self.num_heads, C // self.num_heads])
        kvglobal = tf.reshape(kvglobal, [B, Ntotal, 2, self.num_heads, C // self.num_heads])
        q_avg = tf.reshape(q_avg,[B,1,self.num_heads, C // self.num_heads])



        # Permute to (3, B_, num_heads, N, C // num_heads)
        '''
        3: referring to q, k, v (total 3)
        B: batch size
        num_heads: multi-headed attention
        N:  M^2, referring to each token(patch)
        C // num_heads: Each head of each of (q,k,v) handles C // num_heads -> match exact dimension for multi-headed attention
        '''
        kv = tf.transpose(kv, perm=[2, 0, 3, 1, 4])
        kvglobal = tf.transpose(kvglobal, perm=[2, 0, 3, 1, 4])
        q  = tf.transpose(q, perm=[2, 0, 3, 1, 4])
        q_avg = tf.transpose(q_avg, perm = [0,2,1,3])

        


        # Decompose to query/key/vector for attention
        # each of q, k, v has dimension of (B_, num_heads, N, C // num_heads)
        k, v = kv[0], kv[1] # Why not tuple-unpacking?


        kglobal, vglobal = kvglobal[0], kvglobal[1]
        
        q = q * self.scale

        q = q[0]

        # attn becomes (B_, num_heads, N, N) shape
        # N = M^2

        k = tf.transpose(k, perm=[0, 1, 3, 2])
        kglobal = tf.transpose(kglobal, perm=[0, 1, 3, 2])



        attn = (q @ k)

        attnglobal = (q_avg @ kglobal)
        
        # Remember that relative_position_bias_table = ((2M-1)*(2M-1), num_heads), B_hat from the paper
        # relative_position_index's elements are in range [0, 2M-2]
        # Convert to (M^2, M^2, num_heads). This is B matrix from the paper
        #print("Relative position index", self.relative_position_index.shape)

        a = tf.reshape(self.relative_position_index, [-1])

        b = tf.gather(self.relative_position_bias_table, a)

        a = tf.reshape(b, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1]+1, -1])
        relative_position_bias = a
        # Convert to (num_heads, M^2, M^2) to match the dimension for addition
        relative_position_bias = tf.transpose(relative_position_bias, perm=[2, 0, 1])
        
        # (B, num_heads, N, N) + (1, num_heads, M^2, M^2), where N=M^2
        # attn becomes (B_, num_heads, N, N) or (B, num_heads, M^2, M^2)

       


        attn = attn + tf.expand_dims(relative_position_bias, 0)
        
        if mask is not None:
            nW = mask.shape[0] # nW = number of windows

            
            # attn.view(...) = (B, nW, num_heads, N, N)
            # mask.unsqueeze(1).unsqueeze(0) = (1, num_windows, 1, M^2, M^2)
            # So masking is broadcasted along B and num_heads axis which makes sense


            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N-1, N])  + tf.expand_dims(tf.expand_dims(mask, 1), 0)
            
            # attn = (nW * B, num_heads, N, N)
            attn = tf.reshape(attn, [-1, self.num_heads, N-1, N])
            attn = tf.keras.activations.softmax(attn)
        else:

            attn = tf.keras.activations.softmax(attn)
            

        attnglobal = tf.keras.activations.softmax(attnglobal)

        
        # attn = (nW*B, num_heads, N, N)
        # v = (B_, num_heads, N, C // num_heads). B_ = nW*B
        # attn @ v = (nW*B, num_heads, N, C // num_heads)
        # (attn @ v).transpose(1, 2) = (nW*B, N, num_heads, C // num_heads)
        # Finally, x = (nW*B, N, C), reshape(B_, N, C) performs concatenation of multi-headed attentions
        x = (attn @ v)

        xavg = (attnglobal @ vglobal)

        xavg = tf.reshape(tf.transpose(xavg, perm=[0, 2, 1, 3]), [B, 1, C])

        x = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), [B_, NG, C])
        
        return x, xavg        

class SwinTransformerBlock(tf.keras.layers.Layer):
    """ Swin Transformer Block. It's used as either W-MSA or SW-MSA depending on shift_size
    
    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        num_heads (int): Number of attention heads
        window_size (int): Window size
        shift_size (int): Shift size for SW-MSA
        mlp_ratio (float):Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer(nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): NOrmalization layer. Default: nn.LayerNorm
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=6, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.keras.activations.gelu, norm_layer=tf.keras.layers.LayerNormalization,
                 name=None):
        super(SwinTransformerBlock,self).__init__(name=name)
        self._name = name
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        
        # If window_size > input_resolution, no partition
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(name="normlayer1")

        # Attention
        window_name = name + 'window'
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        name=window_name)
        
        if drop_path > 0.:
            self.drop_path_status = True
        else:
            self.drop_path_status = False

        self.norm2 = norm_layer(name="normlayer2")
        
        Mlp_name = self._name + 'MLP'
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, name=Mlp_name)
        
        # Attention Mask for SW-MSA
        # This handling of attention-mask is my favourite part. What a beautiful implementation.
        if self.shift_size > 0:
            H, W = self.input_resolution
            
            # To match the dimension for window_partition function
            img_mask = np.zeros([1, H, W, 1])
            
            # h_slices and w_slices divide a cyclic-shifted image to 9 regions as shown in the paper
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            
            # Fill out number for each of 9 divided regions
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
                
            img_mask = tf.convert_to_tensor(img_mask)
                    
            mask_windows = window_partition(img_mask, self.window_size) # (nW, M, M, 1)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            
            # Such a gorgeous code..
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)

            tmask1 = (attn_mask == 0)
            attn_mask = tf.where(tmask1, float(0.0), float(-100.0))

        else:
            attn_mask = None
            
        self.attn_mask = attn_mask
        
    
    def call(self, x, training):
        
        H, W = self.input_resolution
        B, L, C = x.shape
        B = tf.shape(x)[0]
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x # Residual
        x = self.norm1(x)
        x = tf.reshape(x, [B, H, W, C]) # H, W refer to the number of "patches" for width and height, not "pixels"
        
        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
        
        # Partition Windows
        x_windows = window_partition(shifted_x, self.window_size) # (nW*B, M, M, C)
        x_windows = tf.reshape(x_windows, [-1, self.window_size*self.window_size, C]) # (nW*B, window_size*window_size, C)
        
        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask) # (nW*B, window_size*window_size, C)
        
        # Merge Windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W) # (B, H', W', C)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = tf.reshape(x, [B, H*W, C])

        # FFn
        x = shortcut + DropPath(x, self.drop_path, training = training)
    
        x = x + DropPath(self.mlp(self.norm2(x)), self.drop_path, training = training)
        # x = x + (self.mlp(self.norm2(x)))
        
        return x



class RefiningEncoderBlock(tf.keras.layers.Layer):
    """ Swin Transformer Block. It's used as either W-MSA or SW-MSA depending on shift_size
    
    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        num_heads (int): Number of attention heads
        window_size (int): Window size
        shift_size (int): Shift size for SW-MSA
        mlp_ratio (float):Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer(nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): NOrmalization layer. Default: nn.LayerNorm
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=6, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=tf.keras.activations.gelu, norm_layer=tf.keras.layers.LayerNormalization,
                 name=None):
        super(RefiningEncoderBlock,self).__init__(name=name)
        self._name = name
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.drop_path = drop_path
        
        # If window_size > input_resolution, no partition
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        normname1 = name+'norm1' 
        self.norm1 = norm_layer(name=normname1)

        # Attention
        window_name = name+'window2'
        self.attn = WindowAttention2(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,name=window_name
        )
        
        if drop_path > 0.:
            self.drop_path_status = True
        else:
            self.drop_path_status = False

        normname2 = name + 'norm2'
        self.norm2 = norm_layer(name=normname2)

        

        
        # Attention Mask for SW-MSA
        # This handling of attention-mask is my favourite part. What a beautiful implementation.
        if self.shift_size > 0:
            H, W = self.input_resolution
            
            # To match the dimension for window_partition function
            img_mask = np.zeros([1, H, W, 1])
            
            # h_slices and w_slices divide a cyclic-shifted image to 9 regions as shown in the paper
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            
            # Fill out number for each of 9 divided regions
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
                
            img_mask = tf.convert_to_tensor(img_mask)


                    
            mask_windows = window_partition(img_mask, self.window_size) # (nW, M, M, 1)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])
            
            # Such a gorgeous code..
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)


            tmask1 = (attn_mask == 0)
            attn_mask = tf.where(tmask1, float(0.0), float(-100.0))
            
            last_col = tf.expand_dims(attn_mask[:,:,35],2)
            last_col = last_col*0.0

            attn_mask = tf.concat([attn_mask,last_col], axis=2)




        else:
            attn_mask = None
            
        self.attn_mask = attn_mask
        
    
    def call(self, x_c, training):
        x = x_c[:,:-1,:]
        xavg = x_c[:,-1,:]
        xavg = tf.expand_dims(xavg,axis=1)

        H, W = self.input_resolution
        B, L, C = x.shape
        B = tf.shape(x)[0]
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x # Residual
        shortcutglobal = xavg
        x = tf.reshape(x, [B, H, W, C]) # H, W refer to the number of "patches" for width and height, not "pixels"
        
        # Cyclic Shift
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x
        
        # Partition Windows
        x_windows = window_partition(shifted_x, self.window_size) # (nW*B, M, M, C)
        x_windows = tf.reshape(x_windows, [-1, self.window_size*self.window_size, C]) # (nW*B, window_size*window_size, C)
        
        # W-MSA / SW-MSA
        attn_windows, xavg = self.attn(x_windows,shortcut,xavg, mask=self.attn_mask) # (nW*B, window_size*window_size, C)
        # Merge Windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W) # (B, H', W', C)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = tf.reshape(x, [B, H*W, C])

        # FFn
        x = shortcut + DropPath(x, self.drop_path, training)

        xavg = shortcutglobal + xavg
   
        x = self.norm1(x)

        xavg = self.norm2(xavg)


        x_c = tf.concat([x,xavg],axis=1)
        return x_c

class PatchMerging(tf.keras.layers.Layer):
    """ Patch Merging Layer from the paper (downsampling)
    Args:
        input_solution (tuple[int]): Resolution of input feature
        dim (int): Number of input channels. (C)
        norm_layer (nn.Module, optional): Normalization layer. (Default: nn.LayerNorm)
    """
    
    def __init__(self, input_resolution, dim, norm_layer=tf.keras.layers.LayerNormalization,name=None):
        super(PatchMerging,self).__init__(name=name)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False)
        self.norm = norm_layer()

        
    def call(self, x):
        """
        x: (B, H*W, C)
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        B = tf.shape(x)[0]
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = tf.reshape(x, [B, H, W, C])
        
        # Separate per patch by 2 x 2
        #t = x.numpy()
        x0 = x[:, 0::2, 0::2, :] # (B, H/2, W/2, C) (top-left of 2x2)
        x1 = x[:, 1::2, 0::2, :] # (B, H/2, W/2, C) (bottom-left of 2x2)
        x2 = x[:, 0::2, 1::2, :] # (B, H/2, W/2, C) (top-right of 2x2)
        x3 = x[:, 1::2, 1::2, :] # (B, H/2, W/2, C) (bottom-right of 2x2)
        
        # Merge by channel -> (B, H/2, W/2, 4C)
        x = tf.concat([x0, x1, x2, x3], -1)

        # Flatten H, W
        x = tf.reshape(x, [B, tf.shape(x)[1]*tf.shape(x)[2], 4 * C])
        x = self.norm(x)

        # Reduction Layer: 4C -> 2C
        x = self.reduction(x)

        return x

class BasicLayer(tf.keras.layers.Layer):
    """ Swin Transformer layer for one stage
    
    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        depth (int): Number of blocks (depending on Swin Version - T, L, ..)
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. (Default: True)
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate (Default: 0.0)
        attn_drop (float, optional): Attention dropout rate (Default: 0.0)
        drop_path (float | tuple[float], optional): Stochastic depth rate (Default: 0.0)
        norm_layer (nn.Module, optional): Normalization layer (Default: nn.LayerNorm)
        downsample (nn.Module | NOne, optional): Downsample layer at the end of the layer (Default: None)
        use_checkpoint (bool): Whether to use checkpointing to save memory (Default: False)
    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=tf.keras.layers.LayerNormalization, downsample=None, use_checkpoint=False, name=None):
        super(BasicLayer,self).__init__(name=name)
        self._name = name
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        
        # Build  Swin-Transformer Blocks
        #self.blocks = []
        #for i in range(depth):
        #    self.blocks.append(            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
        #                         num_heads=num_heads, window_size=window_size,
        #                         shift_size=0 if (i % 2 == 0) else window_size // 2,
        #                         mlp_ratio = mlp_ratio,
        #                         qkv_bias=qkv_bias, qk_scale=qk_scale,
         #                        drop=drop, attn_drop=attn_drop,
        #                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
        #                         norm_layer=norm_layer, name='SWinTransformer' + str(i)
        #                        ))

        #self.blocks = tf.keras.Sequential(self.blocks)

        self.blocks = [SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio = mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, name='SWinTransformer' + str(i)
                                ) for i in range(depth)]

        self.blocks = tf.keras.Sequential(self.blocks)
        
        # Patch Merging Layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer,name="patchmerge"+name)
        else:
            self.downsample = None
            
            
    def call(self, x,training):
        #print("basic layer shape: ", x.shape)
        #print("basic layer input resolution: ", self.input_resolution )
        x = self.blocks(x,training)
            
        
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class RefiningLayer(tf.keras.layers.Layer):
    """ Swin Transformer layer for one stage
    
    Args:
        dim (int): Number of input channels
        input_resolution (tuple[int]): Input resolution
        depth (int): Number of blocks (depending on Swin Version - T, L, ..)
        num_heads (int): Number of attention heads
        window_size (int): Local window size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. (Default: True)
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        drop (float, optional): Dropout rate (Default: 0.0)
        attn_drop (float, optional): Attention dropout rate (Default: 0.0)
        drop_path (float | tuple[float], optional): Stochastic depth rate (Default: 0.0)
        norm_layer (nn.Module, optional): Normalization layer (Default: nn.LayerNorm)
        downsample (nn.Module | NOne, optional): Downsample layer at the end of the layer (Default: None)
        use_checkpoint (bool): Whether to use checkpointing to save memory (Default: False)

    """
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=tf.keras.layers.LayerNormalization, downsample=None, use_checkpoint=False, name=None):
        super(RefiningLayer,self).__init__(name=name)
        self._name = name
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # Build  Swin-Transformer Blocks
        #self.blocks = []
        #for i in range(depth):
         #   self.blocks.append(RefiningEncoderBlock(dim=dim, input_resolution=input_resolution,
         #                        num_heads=num_heads, window_size=window_size,
         #                        shift_size=0 if (i % 2 == 0) else window_size // 2,
         #                        mlp_ratio = mlp_ratio,
         #                        qkv_bias=qkv_bias, qk_scale=qk_scale,
         #                        drop=drop, attn_drop=attn_drop,
         #                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
         #                        norm_layer=norm_layer, name='RefiningEncoderBlock' + str(i)
         #                       ))

        self.blocks = [RefiningEncoderBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio = mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, name='RefiningEncoderBlock' + str(i)
                                ) for i in range(depth)]

        self.blocks = tf.keras.Sequential(self.blocks)

        
        
            
            
    def call(self, x_c, training):
        x_c = self.blocks(x_c,training)
        

        return x_c


class PatchEmbed(tf.keras.layers.Layer):
    """ Convert image to patch embedding
    
    Args:
        img_size (int): Image size (Default: 224)
        patch_size (int): Patch token size (Default: 4)
        in_channels (int): Number of input image channels (Default: 3)
        embed_dim (int): Number of linear projection output channels (Default: 96)
        norm_layer (nn.Module, optional): Normalization layer (Default: None)
    """
    
    def __init__(self, img_size=384, patch_size=4, in_chans=3, embed_dim=64, norm_layer=None,name=None):
        super(PatchEmbed,self).__init__(name=name)
        #self.name=name
        img_size = to_2tuple(img_size) # (img_size, img_size) to_2tuple simply convert t to (t,t)
        patch_size = to_2tuple(patch_size) # (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]] # (num_patches, num_patches)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # proj layer: (B, 3, 224, 224) -> (B, 96, 56, 56)
        Convname = name+'conv2D'
        self.proj = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, name = Convname)
        
        if norm_layer is not None:
            self.norm = norm_layer(name=name+'normpatch')
        else:
            self.norm = None
        
    def call(self, x):
        """
        x: (B, C, H, W) Default: (B, 3, 224, 224)
        returns: (B, H//patch_size * W//patch_size, embed_dim) (B, 56*56, 96)
        """
        B, C, H, W = x.shape
        B = tf.shape(x)[0]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}]) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
 
        # (B, 3, 224, 224) -> (B, 96, 56, 56)
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        x = self.proj(x)
        x = tf.transpose(x, perm=[0, 3, 2, 1])

        # (B, 96, 56, 56) -> (B, 96, 56*56)
        x = tf.reshape(x, [tf.shape(x)[0], x.shape[1], x.shape[2]*x.shape[3]])

        # (B, 96, 56*56) -> (B, 56*56, 96): 56 refers to the number of patches
        x = tf.transpose(x, perm=[0, 2, 1])

        if self.norm is not None:
            x = self.norm(x)

        return x

class PoolConcat(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_first", keepdims=True,name='concat')

    def call(self,x):
        xavg = tf.transpose(x, perm=[0, 2, 1])
        xavg = self.avgpool(xavg) # (B, C, 1)
        xavg = tf.transpose(xavg, perm = [0,2,1])
        x_c = tf.concat([x,xavg],axis=1)

        return x_c

class SwinTransformer(tf.keras.Model):
    """ Swin Transformer
    
    Args:
        img_size (int | tuple(int)): Input image size (Default 224)
        patch_size (int | tuple(int)): Patch size (Default: 4)
        in_chans (int): Number of input image channels (Default: 3)
        num_classes (int): Number of classes for classification head (Default: 1000)
        embed_dim (int): Patch embedding dimension (Default: 96)
        depths (tuple(int)): Depth of each Swin-T layer
        num_heads (tuple(int)): Number of attention heads in different layers
        window_size (int): Window size (Default: 7)
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. (Default: 4)
        qkv_bias (bool): If True, add a learnable bias to query, key, value (Default: True)
        qk_scale (float); Override default qk scale of head_dim ** -0.5 if set. (Default: None)
        drop_rate (float): Dropout rate (Default: 0)
        attn_drop_rate (float): Attention dropout rate (Default: 0)
        drop_path_rate (float); Stochastic depth rate (Default: 0.1)
        norm_layer (nn.Module): Normalization layer (Default: nn.LayerNorm)
        ape (bool): Refers to absolute position embedding. If True, add ape to the patch embedding (Default: False)
        patch_norm (bool): If True, add normalization after patch embedding (Default: True)
        use_checkpoint (bool): Whether to use checkpointing to save memory (Default: False)
    """
    
    def __init__(self, img_size=384, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, depths=[2, 2, 6, 2], num_refining_layers=3,num_heads=[4, 8, 16, 32],
                 window_size=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=tf.keras.layers.LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False,name=None, **kwargs):
        super(SwinTransformer,self).__init__(name=name)
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.num_refining_layers = num_refining_layers
        
        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            name="patch_embed")
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        self.pos_drop = tf.keras.layers.Dropout(drop_rate,name="Dropout1")
        
        # Stochastic Depth
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))] # stochastic depth decay rule
        
        # build layers
        self.basiclayers=[]

        for i_layer in range(self.num_layers):
            self.basiclayers.append(BasicLayer(
            dim=int(embed_dim * 2 ** i_layer),
            input_resolution=(
                patches_resolution[0] // (2 ** i_layer), # After patch-merging layer, patches_resolution(H, W) is halved
                patches_resolution[1] // (2 ** i_layer),
                                ),
            depth=depths[i_layer],
            num_heads=num_heads[i_layer],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
            norm_layer=norm_layer,
            downsample=PatchMerging if (i_layer < self.num_layers -1) else None, # No patch merging at the last stage
            use_checkpoint=use_checkpoint, name='BasicLayer' + str(i_layer)
            ))

        self.basiclayers = tf.keras.Sequential(self.basiclayers)

        self.refining_layers= []


        for i_layer in range(self.num_refining_layers):
            self.refining_layers.append(RefiningLayer(
            dim=int(embed_dim * 2 ** (self.num_layers-1)),
            input_resolution=(
                patches_resolution[0] // (2 ** (self.num_layers-1)), # After patch-merging layer, patches_resolution(H, W) is halved
                patches_resolution[1] // (2 ** (self.num_layers-1)),
                                ),
            depth=2,
            num_heads=4,
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path=0,
            norm_layer=norm_layer,
            downsample= None, # No patch merging at the last stage
            use_checkpoint=use_checkpoint, name='RefiningLayer' + str(i_layer)
            ))

        self.refining_layers = tf.keras.Sequential(self.refining_layers)    
        self.norm = norm_layer()
        self.concatpool = PoolConcat()
        
        # Classification Head
        self.head = tf.keras.layers.Dense(num_classes) if num_classes > 0 else tf.identity

    def build(self, layer):
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel_initializer = TruncatedNormal(stddev=0.02,name=layer.name+'dense')
            if layer.bias is not None:
                layer.bias_initializer = Constant(0,name=layer.name+'bias')
        elif isinstance(layer, tf.keras.layers.LayerNormalization):
            layer.bias_initializer = Constant(0,name=layer.name+'beta')
            layer.gamma_initializer = Constant(1.0,name=layer.name+'gamma')


    def call(self, x, training):
        x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.basiclayers(x,training)

        x = self.norm(x) # (B, L, C)

        x_c = self.concatpool(x)
        #xavg = tf.transpose(x, perm=[0, 2, 1])
        #xavg = self.avgpool(xavg) # (B, C, 1)
        #x = torch.flatten(x, 1)
        #xavg = tf.transpose(xavg, perm = [0,2,1])
        #x_c = tf.concat([x,xavg],axis=1)

        #for refining_layer in self.refining_layers.layers:
            #x,xavg = refining_layer(x, xavg)
        x_c = self.refining_layers(x_c,training)
        x = x_c[:,:-1,:]
        xavg = x_c[:,-1,:]
        xavg = tf.expand_dims(xavg,axis=1)
        return x, xavg

    def summary(self, line_length=None, positions=None, print_fn=None):

        encoder_input = keras.Input(shape=(3,384,384), name="original_img")

        model = SwinTransformer()

        #for i, layer in enumerate(model.layers):

        encoder_output = model(encoder_input)
        Swinmodel = keras.Model(encoder_input, encoder_output, name="SWinModel_Total")

        print(Swinmodel.summary())

    
        encoder_output = self.patch_embed(encoder_input)
        patchembed = keras.Model(encoder_input, encoder_output, name="patchembed")
        #print(patchembed.summary())

        posdrop_input = encoder_output

        posdrop_output = self.pos_drop(posdrop_input)
        posdrop = keras.Model(posdrop_input, posdrop_output, name="pos_drop")
        #print(posdrop.summary())

        merged_summary_str = tf.keras.Sequential()
        merged_summary_str.add(patchembed)
        merged_summary_str.add(posdrop)
        #print(merged_summary_str.summary())

        layer_input = posdrop_output


        for j,layer in enumerate(self.basiclayers.layers):
            for i,SWINlayer in enumerate(layer.blocks.layers):
                layer_output = SWINlayer(layer_input)
                
                layersmodel = keras.Model(layer_input, layer_output, name="BasicLayer_" + str(j) + "_SWINlayer_" + str(i))

                #print(layersmodel.summary())
                merged_summary_str.add(layersmodel)
                layer_input = layer_output
            if layer.downsample is not None:
                    layer_output = layer.downsample(layer_input)
                    downsamplemodel = keras.Model(layer_input, layer_output, name="downsample" + str(j))
                    
                    merged_summary_str.add(downsamplemodel)

                    #print(downsamplemodel.summary())
                    layer_input = layer_output
        
        norm_input = layer_output
        norm_output = self.norm(norm_input)
        norm_layer = keras.Model(norm_input,norm_output,name="Norm_Layer" )
        merged_summary_str.add(norm_layer)

        ####

        #x_c = tf.concat([x,xavg],axis=1)

        #for refining_layer in self.refining_layers.layers:
            #x,xavg = refining_layer(x, xavg)
        #x_c = self.refining_layers(x_c)
        #x = x_c[:,:-1,:]
        #xavg = x_c[:,-1,:]
        
        ###



        merged_input =  self.concatpool(norm_output)
        concatenationlayer = keras.Model(norm_output,merged_input,name='averaging_concatenation_block')
        merged_summary_str.add(concatenationlayer)





        for i,refining_layer in enumerate(self.refining_layers.layers):
   
            merged_output = refining_layer(merged_input)

            refining_layer_model = keras.Model(merged_input,merged_output,name="refiningblock"+str(i))
            merged_summary_str.add(refining_layer_model)
    
            merged_input = merged_output
        

        merged_summary_str._name = "SWinModel_Total_Verbose"
        print(merged_summary_str.summary())

        merged_summary_str = tf.keras.Model(merged_summary_str.input, merged_summary_str.output)

        plot_model(merged_summary_str,to_file='model.png',)




        #SWinModel = keras.Model(encoder_input, output, name="encoder")

        #print(SWinModel.summary())


        #print(self.patch_embed.summary())
    
if __name__ == '__main__':


    dummy = np.random.rand(1, 3, 384, 384)
    model = SwinTransformer()

    model.summary()

    output = model(dummy)
    print(output[0].shape)

    #encoder_input = keras.Input(shape=(3,384,384), name="original_img")

    #windowmodel = BasicLayer( 64, [96,96], 1, 4, 6,
                # mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                # drop_path=0., norm_layer=tf.keras.layers.LayerNormalization, downsample=None, use_checkpoint=False, name='check')

    #dummy = np.random.rand(2,9216,64)

    windowmodel = model




    

    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    optimizer = tf.keras.optimizers.Adam()
    #patchembed = keras.Model(encoder_input, encoder_output, name="patchembed")
    with tf.GradientTape() as tape:
 
        # Run the forward pass of the model to generate a prediction
        output,outputavg = windowmodel(dummy)
        print(output.shape)
        target = tf.random.normal(tf.shape(output))
 
        # Compute the training loss
        loss = mse(target, output)
        print("\nLoss\n", loss)
        # Compute the training accuracy
 
    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, windowmodel.trainable_weights)
 
    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, windowmodel.trainable_weights))



    #output = patchembed(dummy)
    


 