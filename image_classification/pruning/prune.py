# Imports
import numpy as np
import tensorflow as tf

# Settings (Note: Disable UC Pruning in Inference to keep Masks)
enable_uc_pruning = False
threshold = 1.00 # Default: 0.75

# Check Mask
def check_mask(layer):
	# Check Mask Attribute
	if hasattr(layer, 'mask'):
		# Check Mask Usage
		return layer.use_mask
	return False

# Get Layer Name and Type
def get_layer_name_type(layer):
	layer_name = layer.name
	if ("fc" in layer_name) or ("dense" in layer_name):
		layer_type = "Dense"
	elif ("conv" in layer_name) or ("res" in layer_name):
		layer_type = "Conv2D"
	else:
		layer_type = "Unknown"
	return layer_name, layer_type

# Get Layer Weights
def get_layer_weights(layer):
	num_variables = len(layer.get_weights())
	layer_weights = np.array([])
	layer_biases = np.array([])
	layer_mask = np.array([])
	if (num_variables >= 1):
		layer_weights = layer.get_weights()[0]
		if hasattr(layer, 'bias'):
			if layer.use_bias:
				layer_biases = layer.get_weights()[1]
		if hasattr(layer, 'mask'):
			if layer.use_mask:
				layer_mask = layer.get_weights()[-1]
	return num_variables, layer_weights, layer_biases, layer_mask

# Set Layer Weights
def set_layer_weights(layer, weights, biases, mask):
	num_variables = len(layer.get_weights())
	new_weights = []
	if (num_variables >= 1):
		new_weights.append(weights)
		if hasattr(layer, 'bias'):
			if layer.use_bias:
				new_weights.append(biases)
		if hasattr(layer, 'mask'):
			if layer.use_mask:
				new_weights.append(mask)
		layer.set_weights(new_weights)

# UC Pruning of 2D Convolutions
def uc_pruning_conv2d(layer, weights, biases):
	# Invariant Transformations
	abs_weights = np.abs(weights)
	min_array = np.amin(abs_weights, axis=(0, 1, 2))
	print(" Min Array Shape: ", min_array.shape)
	shifted_weights = abs_weights - min_array
	print(" Shifted Array Shape: ", shifted_weights.shape)
	# Get Nodes/Filters Mean
	weights_means = np.mean(shifted_weights, axis=(0, 1, 2))
	print(" Means Array Shape: ", weights_means.shape)
	# Generate Mask
	pruned_by_pca = 0
	pruned_by_threshold = 0
	num_weights_pruned = 0
	kw, kl, inputs, outputs = abs_weights.shape
	mask = np.ones(abs_weights.shape, dtype=np.float32)
	for i in range(kw):
		for j in range(kl):
			for k in range(inputs):
				for l in range(outputs):
					relative_mean = threshold * weights_means[l]
					if (abs_weights[i][j][k][l] < relative_mean):
						mask[i][j][k][l] = 0.0
						pruned_by_threshold += 1
						num_weights_pruned += 1
					if (abs_weights[i][j][k][l] == 0):
						pruned_by_pca += 1
						num_weights_pruned += 1
	print(" Pruned weights by PCA: ", pruned_by_pca)
	print(" Pruned weights by UC: ", pruned_by_threshold)
	# Assign Mask
	set_layer_weights(layer, weights, biases, mask)
	# Count Pruned Weights
	l_num_pruned = num_weights_pruned
	l_total = weights.size + biases.size
	l_pruned = float(l_num_pruned) / l_total * 100.0
	print(" Pruned Weights: {0} / {1} ({2}%)".format(l_num_pruned, l_total, l_pruned))
	return l_num_pruned, l_total

# UC Pruning of FCs
def uc_pruning_dense(layer, weights, biases):
	# Invariant Transformations
	abs_weights = np.abs(weights)
	min_array = np.amin(abs_weights, axis=0)
	print(" Min Array Shape: ", min_array.shape)
	shifted_weights = abs_weights - min_array
	print(" Shifted Array Shape: ", shifted_weights.shape)
	# Get Nodes/Filters Mean
	weights_means = np.mean(shifted_weights, axis=0)
	print(" Means Array Shape: ", weights_means.shape)
	# Generate Mask
	pruned_by_pca = 0
	pruned_by_threshold = 0
	num_weights_pruned = 0
	inputs, outputs = abs_weights.shape
	mask = np.ones(abs_weights.shape, dtype=np.float32)
	for i in range(inputs):
		for j in range(outputs):
			relative_mean = threshold * weights_means[j]
			if (abs_weights[i][j] < relative_mean):
				mask[i][j] = 0.0
				pruned_by_threshold += 1
				num_weights_pruned += 1
			if (abs_weights[i][j] == 0):
				pruned_by_pca += 1
				num_weights_pruned += 1
	print(" Pruned weights by PCA: ", pruned_by_pca)
	print(" Pruned weights by UC: ", pruned_by_threshold)
	# Assign Mask
	set_layer_weights(layer, weights, biases, mask)
	# Count Pruned Weights
	l_num_pruned = num_weights_pruned
	l_total = weights.size + biases.size
	l_pruned = float(l_num_pruned) / l_total * 100.0
	print(" Pruned Weights: {0} / {1} ({2}%)".format(l_num_pruned, l_total, l_pruned))
	return l_num_pruned, l_total

# Check Final Pruning
def check_pruning(model: tf.keras.Model):
	# Counters
	t_num_pruned = 0
	t_total = 0	
	# Iterate over Layers
	for layer in model.layers:
		# Get Weights
		for var in layer.get_weights():
			t_total += var.size
		if hasattr(layer, 'mask'):
			if layer.use_mask:
				layer_mask = layer.get_weights()[-1]
				t_num_pruned += np.count_nonzero(layer_mask==0)
				t_total -= layer_mask.size
	t_pruned = float(t_num_pruned) / t_total * 100.0
	print("Checking Global Pruning: {0} / {1} ({2}%)".format(t_num_pruned, t_total, t_pruned))

# Unimportant Connections (UC) Pruning
def uc_pruning(model: tf.keras.Model):
	# Check UC Pruning Enable
	if enable_uc_pruning:
		# Counters
		t_num_pruned = 0
		t_total = 0
		# Iterate over Layers
		for layer in model.layers:
			# Check Mask Attribute and Usage
			if check_mask(layer):
				# Layer Name and Type
				layer_name, layer_type = get_layer_name_type(layer)
				print("Pruning Layer: ", layer.name)
				print(" Layer Type: ", layer_type)
				# Get the current weights, biases and mask
				num_variables, layer_weights, layer_biases, layer_mask = get_layer_weights(layer)
				print(" Number of Variables:", num_variables)
				print(" Weights Shape:", layer_weights.shape)
				print(" Biases Shape:", layer_biases.shape)
				print(" Mask Shape:", layer_mask.shape)
				# Apply current mask
				layer_weights = layer_weights * layer_mask
				# Prune Weights
				if layer_type == "Conv2D":
					l_num_pruned, l_total = uc_pruning_conv2d(layer, layer_weights, layer_biases)
					t_num_pruned += l_num_pruned
					t_total += l_total
				elif layer_type == "Dense":
					l_num_pruned, l_total = uc_pruning_dense(layer, layer_weights, layer_biases)
					t_num_pruned += l_num_pruned
					t_total += l_total
				else:
					print(" Unknown Layer Type Not Pruned")
		t_pruned = float(t_num_pruned) / t_total * 100.0
		print("Total weights pruned: {0} / {1} ({2}%)".format(t_num_pruned, t_total, t_pruned))
	# Final Pruning Summary
	check_pruning(model)