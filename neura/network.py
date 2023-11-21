from typing import Sequence, List, Tuple
from progress.bar import IncrementalBar

import numpy as np


def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
	return sigmoid(x) * (1 - sigmoid(x))


class Network:
	def __init__(self, weights: Sequence[np.ndarray], biases: Sequence[np.ndarray],
				 learning_rate=1,
				 activation_function=sigmoid,
				 activation_function_derivative=sigmoid_derivative):
		self.num_layers = len(weights)
		self.weights = weights
		self.biases = biases
		self.learning_rate = learning_rate
		self.activation_function = activation_function
		self.activation_function_derivative = activation_function_derivative

	def forward_feed(self, x: np.ndarray) -> np.ndarray:
		"""
		:param x: vector of input activations
		:return: vector of output activations
		"""
		assert x.shape == (self.weights[0].shape[1], 1), "Wrong input vector shape"
		activation = x
		for layer_weights, layer_biases in zip(self.weights, self.biases):
			activation = self.activation_function(np.dot(layer_weights, activation) + layer_biases)
		return activation

	def back_propagation(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
		weights_delta = [np.zeros(w.shape) for w in self.weights]  # Matrix of weights changes
		biases_delta = [np.zeros(b.shape) for b in self.biases]  # Matrix of biases changes

		output_activation = x
		layers_inputs = []
		layers_outputs = []
		for layer_weights, layers_biases in zip(self.weights, self.biases):
			input_activation = np.dot(layer_weights, output_activation) + layers_biases
			layers_inputs.append(input_activation)
			output_activation = self.activation_function(input_activation)
			layers_outputs.append(output_activation)

		error = y - layers_outputs[-1]
		output_error = np.sum(error ** 2)
		for layer in range(self.num_layers - 1, 0, -1):
			weights_delta[layer] = (error * self.activation_function_derivative(
				layers_inputs[layer])) @ layers_outputs[layer - 1].T
			biases_delta[layer] = error
			error = self.learning_rate * np.dot(self.weights[layer].T, error)
		weights_delta[0] = (error * self.activation_function_derivative(layers_inputs[0])) @ x.T
		biases_delta[0] = error

		return weights_delta, biases_delta, output_error

	def fit(self, x_data: Sequence[np.ndarray], y_data: Sequence[np.ndarray], epochs=1):
		bar = IncrementalBar('Learning', max=epochs * len(x_data))
		for e in range(epochs):
			epoch_error = 0
			for (x, y) in zip(x_data, y_data):
				bar.next()
				delta_w, delta_b, error = self.back_propagation(x, y)
				epoch_error += error
				self.weights = [w + dw for w, dw in zip(self.weights, delta_w)]
				self.biases = [b + db for b, db in zip(self.biases, delta_b)]
		bar.finish()


def create_network(structure: Sequence[int], learning_rate=1) -> Network:
	num_layers = len(structure)
	weights = [np.random.randn(structure[layer], structure[layer - 1]) - 0.5 for layer in range(1, num_layers)]
	biases = [np.random.randn(structure[layer], 1) - 0.5 for layer in range(1, num_layers)]
	return Network(weights, biases, learning_rate)
