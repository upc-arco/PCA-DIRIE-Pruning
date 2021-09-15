'''Custom Keras Layers'''

# Imports
import functools
import tensorflow as tf
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops

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

		# Return Outputs
		return outputs

# Custom Conv2D Layer
class MyConv2DLayer(tf.keras.layers.Layer):
	def __init__(self,
				filters,
				kernel_size,
				rank=2,
				strides=(1, 1),
				padding='valid',
				data_format=None,
				dilation_rate=(1, 1),
				groups=1,
				activation=None,
				use_bias=True,
				use_mask=True,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				mask_initializer='ones',
				kernel_regularizer=None,
				bias_regularizer=None,
				trainable=True,
				name=None,
				conv_op=None,
				**kwargs):
		super(MyConv2DLayer, self).__init__(trainable=trainable, name=name, **kwargs)

		# Main Parameters
		self.rank = rank
		if isinstance(filters, float):
			filters = int(filters)
		self.filters = filters
		self.groups = groups or 1
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
		self.activation = tf.keras.activations.get(activation)
		self.use_bias = use_bias
		self.use_mask = use_mask
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.mask_initializer = tf.keras.initializers.get(mask_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self.input_spec = tf.keras.layers.InputSpec(min_ndim=self.rank + 2)
		self._validate_init()
		self._channels_first = self.data_format == 'channels_first'
		self._tf_data_format = conv_utils.convert_data_format(self.data_format, self.rank + 2)

	def _validate_init(self):
		if self.filters is not None and self.filters % self.groups != 0:
			raise ValueError('The number of filters must be evenly divisible by the number of '
							'groups. Received: groups={}, filters={}'.format(self.groups, self.filters))

		if not all(self.kernel_size):
			raise ValueError('The argument `kernel_size` cannot contain 0(s). Received: %s' % (self.kernel_size,))

		if not all(self.strides):
			raise ValueError('The argument `strides` cannot contains 0(s). Received: %s' % (self.strides,))

	def build(self, input_shape):
		# Input/Kernel Shape
		input_shape = tf.TensorShape(input_shape)
		input_channel = self._get_input_channel(input_shape)
		if input_channel % self.groups != 0:
			raise ValueError('The number of input channels must be evenly divisible by the number '
							'of groups. Received groups={}, but the input has {} channels '
							'(full input shape is {}).'.format(self.groups, input_channel, input_shape))
		kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

		# Variables
		self.kernel = self.add_weight(name='kernel',
									shape=kernel_shape,
									initializer=self.kernel_initializer,
									regularizer=self.kernel_regularizer,
									trainable=True,
									dtype=self.dtype)
		if self.use_bias:
			self.bias = self.add_weight(name='bias',
										shape=(self.filters,),
										initializer=self.bias_initializer,
										regularizer=self.bias_regularizer,
										trainable=True,
										dtype=self.dtype)
		else:
			self.bias = None
		if self.use_mask:
			self.mask = self.add_weight(name='mask',
										shape=kernel_shape,
										initializer=self.mask_initializer,
										dtype=self.dtype,
										trainable=False)
		else:
			self.mask = None
		channel_axis = self._get_channel_axis()
		self.input_spec = tf.keras.layers.InputSpec(min_ndim=self.rank + 2, axes={channel_axis: input_channel})

		# Convert Keras formats to TF native formats
		if isinstance(self.padding, str):
			tf_padding = self.padding.upper()
		else:
			tf_padding = self.padding
		tf_dilations = list(self.dilation_rate)
		tf_strides = list(self.strides)

		# Conv2D OP Partial Definition
		self._convolution_op = functools.partial(nn_ops.convolution_v2,
												strides=tf_strides,
												padding=tf_padding,
												dilations=tf_dilations,
												data_format=self._tf_data_format,
												name='Conv2D')
		self.built = True

	def call(self, inputs):
		# Inputs Shape
		input_shape = inputs.shape
		
		# Mask Kernel
		if self.use_mask:
			masked_kernel = self.kernel * self.mask
		else:
			masked_kernel = self.kernel
		
		# Conv2D
		outputs = self._convolution_op(inputs, masked_kernel)

		# Add Bias
		if self.use_bias:
			output_rank = outputs.shape.rank
			# Handle multiple batch dimensions
			if output_rank is not None and output_rank > 2 + self.rank:
				def _apply_fn(o):
					return tf.nn.bias_add(o, self.bias, data_format=self._tf_data_format)
				outputs = conv_utils.squeeze_batch_dims(outputs, _apply_fn, inner_rank=self.rank + 1)
			else:
				outputs = tf.nn.bias_add(outputs, self.bias, data_format=self._tf_data_format)

		# Outputs Shape
		if not tf.executing_eagerly():
			out_shape = self.compute_output_shape(input_shape)
			outputs.set_shape(out_shape)

		# Activation
		if self.activation is not None:
			return self.activation(outputs)

		# Return Outputs
		return outputs

	def _spatial_output_shape(self, spatial_input_shape):
		return [
			conv_utils.conv_output_length(length,
										self.kernel_size[i],
										padding=self.padding,
										stride=self.strides[i],
										dilation=self.dilation_rate[i])
			for i, length in enumerate(spatial_input_shape)
		]

	def compute_output_shape(self, input_shape):
		input_shape = tf.TensorShape(input_shape).as_list()
		batch_rank = len(input_shape) - self.rank - 1
		if self.data_format == 'channels_last':
			return tf.TensorShape(input_shape[:batch_rank] + self._spatial_output_shape(input_shape[batch_rank:-1]) + [self.filters])
		else:
			return tf.TensorShape(input_shape[:batch_rank] + [self.filters] + self._spatial_output_shape(input_shape[batch_rank + 1:]))

	def _get_channel_axis(self):
		if self.data_format == 'channels_first':
			return -1 - self.rank
		else:
			return -1

	def _get_input_channel(self, input_shape):
		channel_axis = self._get_channel_axis()
		if input_shape.dims[channel_axis].value is None:
			raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
		return int(input_shape[channel_axis])
