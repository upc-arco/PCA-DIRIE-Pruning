# Imports
import numpy as np
import pickle
import tensorflow as tf

# Settings
enable_uc_pruning = False
enable_pca_pruning = False
uc_threshold = 0.75 # Default: 0.75

# Check Mask
def check_mask(layer):
	# Check Mask Attribute
	if hasattr(layer, 'mask'):
		# Check Mask Usage
		return layer.use_mask
	elif hasattr(layer, '_mask'):
		# Check Mask Usage
		return layer._use_mask
	return False

# Get Layer Name and Type
def get_layer_name_type(layer):
	layer_name = layer.name + "_" + str(layer.layer_id)
	if ("filter_layer" in layer_name) or ("output_layer" in layer_name):
		layer_type = "Dense"
	elif ("query" in layer_name) or ("key" in layer_name) or ("value" in layer_name):
		layer_type = "DenseEinsum_In"
	elif ("output_transform" in layer_name):
		layer_type = "DenseEinsum_Out"
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
		elif hasattr(layer, '_bias'):
			if layer._use_bias:
				layer_biases = layer.get_weights()[1]				
		if hasattr(layer, 'mask'):
			if layer.use_mask:
				layer_mask = layer.get_weights()[-1]
		elif hasattr(layer, '_mask'):
			if layer._use_mask:
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
		elif hasattr(layer, '_bias'):
			if layer._use_bias:
				new_weights.append(biases)
		if hasattr(layer, 'mask'):
			if layer.use_mask:
				new_weights.append(mask)
		elif hasattr(layer, '_mask'):
			if layer._use_mask:
				new_weights.append(mask)
		layer.set_weights(new_weights)

# Get Layers to Prune
def get_layers(model: tf.keras.Model):
	layers_list = []
	# Iterate
	for model_layer in model.layers:
		# Check Transformer Model
		if ("transformer" in model_layer.name):
			for sub_layer in model_layer.layers:
				# Check "layers" Attribute
				if hasattr(sub_layer, 'layers'):
					for sub_sub_layer in sub_layer.layers:
						for layer in sub_sub_layer:
							# Check "layer" Attribute
							if hasattr(layer, 'layer'):
								if hasattr(layer.layer, 'query_dense_layer'):
									layers_list.append(layer.layer.query_dense_layer)
								if hasattr(layer.layer, 'key_dense_layer'):
									layers_list.append(layer.layer.key_dense_layer)
								if hasattr(layer.layer, 'value_dense_layer'):
									layers_list.append(layer.layer.value_dense_layer)
								if hasattr(layer.layer, 'output_dense_layer'):
									layers_list.append(layer.layer.output_dense_layer)
								if hasattr(layer.layer, 'filter_dense_layer'):
									layers_list.append(layer.layer.filter_dense_layer)
	print("Layers to Prune: ", len(layers_list))
	return layers_list

# Check Final Pruning
def check_pruning(model: tf.keras.Model):
	# Counters
	t_num_pruned = 0
	t_total = 0
	# Iterate over Weight Variables
	for weight_var in model.weights:
		# Count Total and Pruned Weights by Mask/Bias
		if ("mask" in weight_var.name):
			layer_mask = weight_var.numpy()
			t_num_pruned += np.count_nonzero(layer_mask==0)
		elif ("bias" in weight_var.name):
			layer_biases = weight_var.numpy()
			t_num_pruned += np.count_nonzero(layer_biases==0)
			t_total += layer_biases.size
		else:
			layer_weights = weight_var.numpy()
			t_total += layer_weights.size
	t_pruned = float(t_num_pruned) / t_total * 100.0
	print("Checking Global Pruning: {0} / {1} ({2}%)".format(t_num_pruned, t_total, t_pruned))

# UC Pruning of FCs
def uc_pruning_dense(layer, weights, biases, layer_type):
	# DenseEinsum 2D Reshapes
	if (layer_type == "DenseEinsum_In"):
		hidden_size, num_heads, dim_per_head = weights.shape
		weights_rs = np.reshape(weights, (hidden_size, num_heads * dim_per_head))
	elif (layer_type == "DenseEinsum_Out"):
		num_heads, dim_per_head, hidden_size = weights.shape
		weights_rs = np.reshape(weights, (num_heads * dim_per_head, hidden_size))
	else:
		weights_rs = weights
	# Invariant Transformations
	abs_weights = np.abs(weights_rs)
	min_array = np.amin(abs_weights, axis=0)
	shifted_weights = abs_weights - min_array
	# Get Nodes/Filters Mean
	weights_means = np.mean(shifted_weights, axis=0)
	# Generate Mask
	pruned_by_pca = 0
	pruned_by_threshold = 0
	num_weights_pruned = 0
	inputs, outputs = abs_weights.shape
	mask = np.ones(abs_weights.shape, dtype=np.float32)
	for i in range(inputs):
		for j in range(outputs):
			relative_mean = uc_threshold * weights_means[j]
			if (abs_weights[i][j] == 0):
				mask[i][j] = 0.0
				pruned_by_pca += 1
				num_weights_pruned += 1
			elif (abs_weights[i][j] < relative_mean):
				mask[i][j] = 0.0
				pruned_by_threshold += 1
				num_weights_pruned += 1
	print(" Pruned weights by PCA: ", pruned_by_pca)
	print(" Pruned weights by UC: ", pruned_by_threshold)
	# Reshape Mask
	if (layer_type == "DenseEinsum_In"):
		mask_rs = np.reshape(mask, (hidden_size, num_heads, dim_per_head))
	elif (layer_type == "DenseEinsum_Out"):
		mask_rs = np.reshape(mask, (num_heads, dim_per_head, hidden_size))
	else:
		mask_rs = mask
	# Assign Mask
	set_layer_weights(layer, weights, biases, mask_rs)
	# Count Pruned Weights
	l_num_pruned = num_weights_pruned
	l_total = weights.size + biases.size
	l_pruned = float(l_num_pruned) / l_total * 100.0
	print(" Pruned Weights: {0} / {1} ({2}%)".format(l_num_pruned, l_total, l_pruned))
	return l_num_pruned, l_total

# Unimportant Connections (UC) Pruning
def uc_pruning(model: tf.keras.Model):
	# Check UC Pruning Enable
	if enable_uc_pruning:
		# Counters
		t_num_pruned = 0
		t_total = 0
		# Get Layers to Prune
		layers_list = get_layers(model)
		# Iterate over Layers
		for layer in layers_list:
			# Check Mask Attribute and Usage
			if check_mask(layer):
				# Layer Name and Type
				layer_name, layer_type = get_layer_name_type(layer)
				print("Pruning Layer: ", layer_name)
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
				if ("Dense" in layer_type):
					l_num_pruned, l_total = uc_pruning_dense(layer, layer_weights, layer_biases, layer_type)
					t_num_pruned += l_num_pruned
					t_total += l_total
				else:
					print(" Unknown Layer Type Not Pruned")
		t_pruned = float(t_num_pruned) / t_total * 100.0
		print("Total Weights Pruned: {0} / {1} ({2}%)".format(t_num_pruned, t_total, t_pruned))
	# Final Pruning Summary
	check_pruning(model)

# PCA Pruning of FCs
def pca_pruning_dense(pca_data, layer, weights, biases, enable_reshape):
	# Get PCA Data
	num_nodes = pca_data[0]
	non_pruned_nodes = pca_data[1]
	nodes_pruned = num_nodes - non_pruned_nodes
	# Generate Mask
	l_num_pruned_w = 0
	if (nodes_pruned > 0):
		idx_prune = np.arange(num_nodes)
		np.random.seed(nodes_pruned)
		np.random.shuffle(idx_prune)
		idx_prune = idx_prune[:nodes_pruned]
		mask = np.ones(num_nodes, dtype=np.float32)
		mask[idx_prune] = 0.0
		if enable_reshape:
			hidden_size, num_heads, dim_per_head = weights.shape
			mask = np.reshape(mask, (num_heads, dim_per_head))
		if (biases.size > 0):
			new_biases = mask * biases
		else:
			new_biases = biases
		mask = mask * np.ones_like(weights)
		l_num_pruned_w = np.count_nonzero(mask==0) + np.count_nonzero(new_biases==0)
		# Assign Mask
		set_layer_weights(layer, weights, new_biases, mask)
	# Count Pruned Nodes and Weights
	l_num_pruned_n = nodes_pruned
	l_total_n = num_nodes
	l_pruned_n = float(l_num_pruned_n) / l_total_n * 100.0
	print(" Pruned Nodes: {0} / {1} ({2}%)".format(l_num_pruned_n, l_total_n, l_pruned_n))
	l_total_w = weights.size + biases.size
	l_pruned_w = float(l_num_pruned_w) / l_total_w * 100.0
	print(" Pruned Weights: {0} / {1} ({2}%)".format(l_num_pruned_w, l_total_w, l_pruned_w))
	return l_num_pruned_n, l_total_n, l_num_pruned_w, l_total_w

# PCA Pruning
def pca_pruning(model: tf.keras.Model):
	# Check PCA Pruning Enable
	if enable_pca_pruning:
		# Counters
		t_num_pruned_n = 0
		t_total_n = 0
		t_num_pruned_w = 0
		t_total_w = 0
		# Load PCA Data
		file = open("pruning/pca_data.pkl", "rb")
		pca_data = pickle.load(file)
		file.close()
		# Get Layers to Prune
		layers_list = get_layers(model)
		# Iterate over Layers
		for layer in layers_list:
			# Check Mask Attribute and Usage
			if check_mask(layer):
				# Layer Name and Type
				layer_name, layer_type = get_layer_name_type(layer)
				print("Pruning Layer: ", layer_name)
				print(" Layer Type: ", layer_type)
				# Get the current weights, biases and mask
				num_variables, layer_weights, layer_biases, layer_mask = get_layer_weights(layer)
				print(" Number of Variables:", num_variables)
				print(" Weights Shape:", layer_weights.shape)
				print(" Biases Shape:", layer_biases.shape)
				print(" Mask Shape:", layer_mask.shape)
				# Prune Nodes
				if (layer_type == "Dense") or ("DenseEinsum" in layer_type):
					enable_reshape = (layer_type == "DenseEinsum_In")
					l_num_pruned_n, l_total_n, l_num_pruned_w, l_total_w = pca_pruning_dense(pca_data[layer_name], layer, 
																							layer_weights, layer_biases, 
																							enable_reshape)
					t_num_pruned_n += l_num_pruned_n
					t_total_n += l_total_n
					t_num_pruned_w += l_num_pruned_w
					t_total_w += l_total_w
				else:
					print(" Unknown Layer Type Not Pruned")
		t_pruned_n = float(t_num_pruned_n) / t_total_n * 100.0
		t_pruned_w = float(t_num_pruned_w) / t_total_w * 100.0
		print("Total Nodes Pruned: {0} / {1} ({2}%)".format(t_num_pruned_n, t_total_n, t_pruned_n))
		print("Total Weights Pruned: {0} / {1} ({2}%)".format(t_num_pruned_w, t_total_w, t_pruned_w))
		# Final Pruning Summary
		check_pruning(model)
