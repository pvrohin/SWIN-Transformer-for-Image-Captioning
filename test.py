from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from transformer import TransformerModel
from time import *
from tensorflow import cast, int32, float32
import tensorflow as tf




# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack
 
# Define the training parameters
epochs = 2
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1
 

dec_vocab_size = 8918
dec_seq_length =  38
enc_vocab_size = 8918
enc_seq_length = 38

training_model = TransformerModel(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

print(training_model.build(False).summary())
 
 
# # Define the model parameters
# h = 8  # Number of self-attention heads
# d_k = 64  # Dimensionality of the linearly projected queries and keys
# d_v = 64  # Dimensionality of the linearly projected values
# d_model = 512  # Dimensionality of model layers' outputs
# d_ff = 2048  # Dimensionality of the inner fully connected layer
# n = 6  # Number of layers in the encoder stack
 
# # Define the training parameters
# epochs = 2
# batch_size = 64
# beta_1 = 0.9
# beta_2 = 0.98
# epsilon = 1e-9
# dropout_rate = 0.1
 

# from numpy import random

 
# # Implementing a learning rate scheduler
# class LRScheduler(LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000, **kwargs):
#         super(LRScheduler, self).__init__(**kwargs)
 
#         self.d_model = cast(d_model, float32)
#         self.warmup_steps = warmup_steps
 
#     def __call__(self, step_num):
 
#         # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
#         arg1 = step_num ** -0.5
#         arg2 = step_num * (self.warmup_steps ** -1.5)
 
#         return (self.d_model ** -0.5) * math.minimum(arg1, arg2)
 
 
# # Instantiate an Adam optimizer
# optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
 
# # Create model
# dec_vocab_size = 8918
# dec_seq_length =  38
# enc_vocab_size = 8918
# enc_seq_length = 37

# training_model = TransformerModel(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
 
 
# # Defining the loss function
# def loss_fcn(target, prediction):
#     # Create mask so that the zero padding values are not included in the computation of loss
#     padding_mask = math.logical_not(equal(target, 0))
#     padding_mask = cast(padding_mask, float32)
 
#     # Compute a sparse categorical cross-entropy loss on the unmasked values
#     loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
 
#     # Compute the mean loss over the unmasked values
#     return reduce_sum(loss) / reduce_sum(padding_mask)
 
 
# # Defining the accuracy function
# def accuracy_fcn(target, prediction):
#     # Create mask so that the zero padding values are not included in the computation of accuracy
#     padding_mask = math.logical_not(equal(target, 0))
 
#     # Find equal prediction and target values, and apply the padding mask
#     print(target.dtype)
#     maxpred = cast(argmax(prediction, axis=2),int32)
#     accuracy = equal(target, maxpred)
#     accuracy = math.logical_and(padding_mask, accuracy)
 
#     # Cast the True/False values to 32-bit-precision floating-point numbers
#     padding_mask = cast(padding_mask, float32)
#     accuracy = cast(accuracy, float32)
 
#     # Compute the mean accuracy over the unmasked values
#     return reduce_sum(accuracy) / reduce_sum(padding_mask)
 
 
# # Include metrics monitoring
# train_loss = Mean(name='train_loss')
# train_accuracy = Mean(name='train_accuracy')
 
# # Create a checkpoint object and manager to manage multiple checkpoints
# ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
# ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)
 
# # Speeding up the training process
# #@function
# def train_step(encoder_input, decoder_input, decoder_output):
#     with GradientTape() as tape:
 
#         # Run the forward pass of the model to generate a prediction
#         prediction = training_model(encoder_input, decoder_input, training=True)
#         print(prediction.shape)
 
#         # Compute the training loss
#         loss = loss_fcn(decoder_output, prediction)
 
#         # Compute the training accuracy
#         #accuracy = accuracy_fcn(decoder_output, prediction)
 
#     # Retrieve gradients of the trainable variables with respect to the training loss
#     gradients = tape.gradient(loss, training_model.trainable_weights)
 
#     # Update the values of the trainable variables by gradient descent
#     optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
 
#     #train_loss(loss)
#     #train_accuracy(accuracy)
 
 
# for epoch in range(epochs):
 
#     train_loss.reset_states()
#     train_accuracy.reset_states()
 
#     print("\nStart of epoch %d" % (epoch + 1))
 

 
#     # Iterate over the dataset batches
#     for i in range(10):
 
#         # Define the encoder and decoder inputs, and the decoder output
#         #encoder_input = train_batchX[:, 1:]
#         train_batchY = cast(tf.convert_to_tensor((random.random([1,38])),int32),dtype=int32)
#         decoder_input = cast(train_batchY[:, :-1],int32)
#         print(f" decoder input shape: {decoder_input.shape}")
#         encoder_input = cast(tf.convert_to_tensor((random.random([1,3,384,384])),float32),dtype=float32)
#         decoder_output = cast(train_batchY[:, 1:],int32)

        
        


 
#         train_step(encoder_input, decoder_input, decoder_output)
 
#         #if step % 50 == 0:
#            # print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
#             # print("Samples so far: %s" % ((step + 1) * batch_size))
 
#     # Print epoch number and loss value at the end of every epoch
#     print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
 
#     # Save a checkpoint after every five epochs
#     if (epoch + 1) % 5 == 0:
#         save_path = ckpt_manager.save()
#         print("Saved checkpoint at epoch %d" % (epoch + 1))
 
# #print("Total time taken: %.2fs" % (time() - start_time))