from SWINblock import *
from dataloader import *
from transformer import TransformerModel
import tensorflow as tf
from tensorflow import cast, float32, int32



train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')
val_loss = Mean(name='val_loss')


# Defining the loss function
#@tf.function
def loss_fcn(target, prediction):
	# Create mask so that the zero padding values are not included in the computation of loss
	padding_mask = math.logical_not(equal(target, 0))
	padding_mask = cast(padding_mask, float32)
 
	# Compute a sparse categorical cross-entropy loss on the unmasked values
	loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
 
	# Compute the mean loss over the unmasked values
	return reduce_sum(loss) / reduce_sum(padding_mask)

# Defining the accuracy function
#@tf.function
def accuracy_fcn(target, prediction):
	# Create mask so that the zero padding values are not included in the computation of accuracy
	padding_mask = math.logical_not(equal(target, 0))
 
	# Find equal prediction and target values, and apply the padding mask
	maxpred = cast(argmax(prediction, axis=2),int32)
	accuracy = equal(target, maxpred)
	accuracy = math.logical_and(padding_mask, accuracy)
 
	# Cast the True/False values to 32-bit-precision floating-point numbers
	padding_mask = cast(padding_mask, float32)
	accuracy = cast(accuracy, float32)
 
	# Compute the mean accuracy over the unmasked values
	return reduce_sum(accuracy) / reduce_sum(padding_mask)

# Speeding up the training process
#@tf.function
def train_step(inputs, decoder_output, training_model, optimizer, train_loss, train_accuracy):
	with GradientTape() as tape:
 
		# Run the forward pass of the model to generate a prediction
		prediction = training_model(inputs, training=True)
		#print(prediction.shape)
 
		# Compute the training loss
		loss = loss_fcn(decoder_output, prediction)
		# Compute the training accuracy
		accuracy = accuracy_fcn(decoder_output, prediction)
 
	# Retrieve gradients of the trainable variables with respect to the training loss
	gradients = tape.gradient(loss, training_model.trainable_weights)
 
	# Update the values of the trainable variables by gradient descent
	optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
 
	train_loss(loss)
	train_accuracy(accuracy)

# Implementing a learning rate scheduler
class LRScheduler(LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000, **kwargs):
		super(LRScheduler, self).__init__(**kwargs)
 
		self.d_model = cast(d_model, float32)
		self.warmup_steps = warmup_steps
 
	def __call__(self, step_num):
 
		# Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
		arg1 = step_num ** -0.5
		arg2 = step_num * (self.warmup_steps ** -1.5)
 
		return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

	def get_config(self):
		config = {
		'd_model': self.d_model,
		'warmup_steps': self.warmup_steps,
	}
		return config




if __name__ == '__main__':
	# Test to load some images alongside its 5 corresponding captions

	train_loss_dict = {}
	val_loss_dict = {}

	images_dir = 'Dataset/Flicker8k_Dataset'

	captions_dir = 'Dataset/Flickr8k_text/Flickr8k.token.txt'

	train_dataset, val_dataset, test_dataset = ddd(images_dir, captions_dir, 1, 10)

	# Define the model parameters
	h = 8  # Number of self-attention heads
	d_k = 64  # Dimensionality of the linearly projected queries and keys
	d_v = 64  # Dimensionality of the linearly projected values
	d_model = 512  # Dimensionality of model layers' outputs
	d_ff = 2048  # Dimensionality of the inner fully connected layer
	n = 6  # Number of layers in the encoder stack
	
	# Define the training parameters
	epochs = 10
	batch_size = 1
	beta_1 = 0.9
	beta_2 = 0.98
	epsilon = 1e-5
	dropout_rate = 0.1

	# Instantiate an Adam optimizer
	optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)
	
	# Create model
	dec_vocab_size = 8918
	dec_seq_length = 39
	enc_vocab_size = 8918
	enc_seq_length = 39

	training_model = TransformerModel(dec_vocab_size, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate,name='SWINtransformer')#.build_graph(False)
	training_model.compile(loss=loss_fcn, optimizer=optimizer)


	pbar = tqdm(enumerate(train_dataset))
	for epoch in (range(epochs)):
	
		train_loss.reset_states()
		train_accuracy.reset_states()
		val_loss.reset_states()

		# Iterate over the dataset batches
		for (step, (train_batchX, train_batchY)) in pbar:

			
			train_batchX = tf.divide(train_batchX, 255.0)
			encoder_input = train_batchX

			decoder_input = cast(train_batchY[:, :-1], int32)
			decoder_output = cast(train_batchY[:, 1:], int32)

			
			
			inputs = [encoder_input,decoder_input]

	
			train_step(inputs, decoder_output, training_model, optimizer, train_loss, train_accuracy)
			pbar.set_postfix({'Epoch, Step, Loss, Accuracy ': [epoch + 1,step,train_loss.result().numpy(),train_accuracy.result().numpy() ]})	
			if (step+1) % 50 == 0:
				pass
				#tqdm.write("Saved checkpoint at epoch %d" % (epoch + 1))
				#print(training_model.save_spec() is None)
				#training_model.save_weights('./checkpoints/my_checkpoint')


		for valstep, (val_batchX, val_batchY) in enumerate(val_dataset):

			# Define the encoder and decoder inputs, and the decoder output
			val_batchX = tf.divide(val_batchX, 255.0)
			decoder_input = val_batchY[:, :-1]
			decoder_output = val_batchY[:, 1:]

			# Generate a prediction
			inputs = [val_batchX,decoder_input]
			prediction = training_model(inputs, training=False)


			# Compute the validation loss
			loss = loss_fcn(decoder_output, prediction)
			val_loss(loss)

	
		# Save a checkpoint after every five epochs
		if (epoch + 1) % 5 == 0:
			training_model.save_weights('./checkpoints/my_checkpoint')
			tqdm.write("Saved checkpoint at epoch %d" % (epoch + 1))
			train_loss_dict[epoch] = train_loss.result()
			val_loss_dict[epoch] = val_loss.result()



	with open('./train_loss.pkl', 'wb') as file:
		dump(train_loss_dict, file)

# Save the validation loss values
	with open('./val_loss.pkl', 'wb') as file:
		dump(val_loss_dict, file)