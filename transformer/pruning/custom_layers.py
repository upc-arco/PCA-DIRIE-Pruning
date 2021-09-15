'''Custom Keras Layers'''

# Imports
import tensorflow as tf
import numpy as np
import pickle
from sklearn.decomposition import PCA

# DenseEinsum Character Indices
_CHR_IDX = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]

# PCA Settings
enable_pca = False
pca_target_it = 75 # Default: 25
pca_threshold = 0.95 # Default: 0.95

# Global Variables
outputs_list = {}
count_iterations = {}
layer_id_counter = 0

# PCA Trace and Analysis
def print_pca_analysis():
	# Check PCA Enable
	if (enable_pca == True):
		# Iterate over each Layer
		num_components_dict = {}
		for key, it in count_iterations.items():
			# Basic Information
			print("Running PCA of Layer: ", str(key))
			print(" Total Iterations: ", it)

			# Shapes
			scores_list = outputs_list[key]
			scores_list_shape = scores_list[0].shape
			print(" Outputs Shape: ", scores_list_shape)

			# Reshapes
			scores_list_np = np.array(scores_list)
			if (len(scores_list_shape) == 4):
				trace_it, batch_size, length, num_heads, dim_per_head = scores_list_np.shape
				scores_list_npr = np.reshape(scores_list_np, (trace_it*batch_size*length, num_heads*dim_per_head))
			elif (len(scores_list_shape) == 3):
				trace_it, batch_size, length, hidden_size = scores_list_np.shape
				scores_list_npr = np.reshape(scores_list_np, (trace_it*batch_size*length, hidden_size))
			else:
				raise ValueError(" Invalid Shape: ", scores_list_shape)
			print(" Scores Shape: ", scores_list_npr.shape)
			print("  Trace Iterations: ", trace_it)
			num_nodes = scores_list_npr.shape[-1]
			print("  Number of Nodes: ", num_nodes)

			# PCA
			print(" PCA Variance: ", pca_threshold)
			pca_trh = PCA(svd_solver='full', n_components=pca_threshold, copy=False)
			scores_pca_trh = pca_trh.fit_transform(scores_list_npr)
			num_components = pca_trh.n_components_
			num_components_dict[key] = [num_nodes, num_components]
			print("  Number of Components: ", num_components)
			print("  Transformed Shape: ", scores_pca_trh.shape)

		# Save Dictionary
		file = open("pruning/pca_data.pkl", "wb")
		pickle.dump(num_components_dict, file)
		file.close()

def pca_trace(layer_id, layer_name, outputs):
	# Key Conversion
	key = str(layer_name, 'utf-8') + "_" + str(layer_id.item())

	# Update Trace
	if key in count_iterations:
		count_iterations[key] += 1
		if (count_iterations[key] <= pca_target_it):
			outputs_list[key].append(outputs)
	else:
		count_iterations[key] = 1
		outputs_list[key] = [outputs]

	# Return Boolean
	return True

# Get Unique Layer ID
def get_layer_id():
	global layer_id_counter
	layer_id_counter += 1
	return layer_id_counter

# Custom Dense Layer
class MyDenseLayer(tf.keras.layers.Layer):
	def __init__(self, 
				units,
				activation=None,
				use_bias=True,
				use_mask=True,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				mask_initializer='ones',
				kernel_regularizer=None,
				bias_regularizer=None,
				**kwargs):
		super(MyDenseLayer, self).__init__(**kwargs)

		# Main Parameters
		self.units = int(units) if not isinstance(units, int) else units
		self.activation = tf.keras.activations.get(activation)
		self.use_bias = use_bias
		self.use_mask = use_mask
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.mask_initializer = tf.keras.initializers.get(mask_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

		# Other Parameters
		self.layer_id = get_layer_id()
		self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
		self.supports_masking = True

	def build(self, input_shape):
		# Data Type
		dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
		if not (dtype.is_floating or dtype.is_complex):
			raise TypeError('Unable to build `Dense` layer with non-floating point ''dtype %s' % (dtype,))

		# Input Shape
		input_shape = tf.TensorShape(input_shape)
		last_dim = tf.compat.dimension_value(input_shape[-1])
		if last_dim is None:
			raise ValueError('The last dimension of the inputs to `Dense` ''should be defined. Found `None`.')
		self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})

		# Variables
		self.kernel = self.add_weight('kernel',
									shape=[last_dim, self.units],
									initializer=self.kernel_initializer,
									regularizer=self.kernel_regularizer,
									dtype=self.dtype,
									trainable=True)
		if self.use_bias:
			self.bias = self.add_weight('bias',
										shape=[self.units,],
										initializer=self.bias_initializer,
										regularizer=self.bias_regularizer,
										dtype=self.dtype,
										trainable=True)
		else:
			self.bias = None
		if self.use_mask:
			self.mask = self.add_weight('mask',
										shape=[last_dim, self.units],
										initializer=self.mask_initializer,
										dtype=self.dtype,
										trainable=False)
		else:
			self.mask = None
		self.built = True

	def call(self, inputs):
		# Inputs DType
		if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
			inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

		# Rank
		rank = inputs.shape.rank

		# Mask Kernel
		if self.use_mask:
			masked_kernel = self.kernel * self.mask
		else:
			masked_kernel = self.kernel

		# MatMul
		if rank == 2 or rank is None:
			if isinstance(inputs, tf.sparse.SparseTensor):
				inputs, _ = tf.sparse.fill_empty_rows(inputs, 0)
				ids = tf.sparse.SparseTensor(indices=inputs.indices, values=inputs.indices[:, 1], dense_shape=inputs.dense_shape)
				weights = inputs
				outputs = tf.nn.embedding_lookup_sparse(masked_kernel, ids, weights, combiner='sum')
			else:
				outputs = tf.matmul(a=inputs, b=masked_kernel)
		else:
			outputs = tf.tensordot(inputs, masked_kernel, [[rank - 1], [0]])
			if not tf.executing_eagerly():
				shape = inputs.shape.as_list()
				output_shape = shape[:-1] + [masked_kernel.shape[-1]]
				outputs.set_shape(output_shape)

		# Add Bias
		if self.use_bias:
			outputs = tf.nn.bias_add(outputs, self.bias)

		# Activation
		if self.activation is not None:
			outputs = self.activation(outputs)

		# PCA
		if ((enable_pca == True) and (self.use_mask == True)):
			pca_op = tf.numpy_function(pca_trace, [self.layer_id, self.name, outputs], tf.bool)
		else:
			pca_op = tf.constant(False)

		# Return Outputs
		with tf.control_dependencies([pca_op]):
			return outputs

# Custom DenseEinsum Layer
@tf.keras.utils.register_keras_serializable(package="Text")
class MyDenseEinsum(tf.keras.layers.Layer):
	def __init__(self,
				output_shape,
				num_summed_dimensions=1,
				activation=None,
				use_bias=True,
				use_mask=True,
				kernel_initializer="glorot_uniform",
				bias_initializer="zeros",
				mask_initializer='ones',
				kernel_regularizer=None,
				bias_regularizer=None,
				**kwargs):
		super(MyDenseEinsum, self).__init__(**kwargs)

		# Main Parameters
		self._output_shape = output_shape if isinstance(output_shape, (list, tuple)) else (output_shape,)
		self._activation = tf.keras.activations.get(activation)
		self._use_bias = use_bias
		self._use_mask = use_mask
		self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self._bias_initializer = tf.keras.initializers.get(bias_initializer)
		self._mask_initializer = tf.keras.initializers.get(mask_initializer)
		self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self._num_summed_dimensions = num_summed_dimensions

		# Other Parameters
		self._einsum_string = None
		self.layer_id = get_layer_id()

	def _build_einsum_string(self, free_input_dims, bound_dims, output_dims):
		input_str = ""
		kernel_str = ""
		output_str = ""
		letter_offset = 0
		for i in range(free_input_dims):
			char = _CHR_IDX[i + letter_offset]
			input_str += char
			output_str += char
		letter_offset += free_input_dims
		for i in range(bound_dims):
			char = _CHR_IDX[i + letter_offset]
			input_str += char
			kernel_str += char
		letter_offset += bound_dims
		for i in range(output_dims):
			char = _CHR_IDX[i + letter_offset]
			kernel_str += char
			output_str += char
		return input_str + "," + kernel_str + "->" + output_str

	def build(self, input_shape):
		# Input/Output Shape/Dimensions
		input_shape = tf.TensorShape(input_shape)
		input_rank = input_shape.rank
		free_input_dims = input_rank - self._num_summed_dimensions
		output_dims = len(self._output_shape)

		# Einstein Summation String
		self._einsum_string = self._build_einsum_string(free_input_dims, self._num_summed_dimensions, output_dims)

		# Kernel Shape
		self._kernel_shape = (input_shape[free_input_dims:].concatenate(self._output_shape))

		# Variables
		self._kernel = self.add_weight("kernel",
									shape=self._kernel_shape,
									initializer=self._kernel_initializer,
									regularizer=self._kernel_regularizer,
									dtype=self.dtype,
									trainable=True)
		if self._use_bias:
			self._bias = self.add_weight("bias",
										shape=self._output_shape,
										initializer=self._bias_initializer,
										regularizer=self._bias_regularizer,
										dtype=self.dtype,
										trainable=True)
		else:
			self._bias = None
		if self._use_mask:
			self._mask = self.add_weight('mask',
										shape=self._kernel_shape,
										initializer=self._mask_initializer,
										dtype=self.dtype,
										trainable=False)
		else:
			self._mask = None
		super(MyDenseEinsum, self).build(input_shape)

	def call(self, inputs):
		# Mask Kernel
		if self._use_mask:
			masked_kernel = self._kernel * self._mask
		else:
			masked_kernel = self._kernel

		# Einstein Summation
		ret = tf.einsum(self._einsum_string, inputs, masked_kernel)

		# Add Bias
		if self._use_bias:
			ret += self._bias

		# Activation
		if self._activation is not None:
			ret = self._activation(ret)

		# PCA
		if ((enable_pca == True) and (self._use_mask == True)):
			pca_op = tf.numpy_function(pca_trace, [self.layer_id, self.name, ret], tf.bool)
		else:
			pca_op = tf.constant(False)

		# Return Outputs
		with tf.control_dependencies([pca_op]):
			return ret
