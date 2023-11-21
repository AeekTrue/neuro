import imageio.v2 as imageio
import numpy as np
from progress.bar import IncrementalBar

import neura


def image2pixels_set(image: np.ndarray) -> np.ndarray:
	if len(image.shape) == 3:
		return image.reshape((image.shape[0] * image.shape[1], image.shape[2], 1))
	elif len(image.shape) == 2:  # Grayscale
		return image.reshape((image.shape[0] * image.shape[1], 1, 1))
	else:
		raise ValueError('Unknown image shape')


def act(x):
	return (x > 1) * 1 + 0.01 * ((0 <= x) * (x <= 1)) * (x - (x > 1))


def act_der(x):
	return 0.01 + 0.99 * ((0 <= x) * (x <= 1))


x_data = image2pixels_set(imageio.imread('sch.jpg')) / 255
y_data = image2pixels_set(imageio.imread('schneg.jpg')) / 255

x_test = image2pixels_set(imageio.imread('chelsea-origin.jpg')) / 255

network = neura.create_network([3, 3, 3], learning_rate=0.3)
# network.activation_function = act
# network.activation_function_derivative = act_der

y_test = np.array([network.forward_feed(x) for x in x_test]).reshape((300, 451, 3)) * 255
imageio.imwrite('result.jpg', y_test.astype(np.uint8))

network.fit(x_data, y_data, epochs=1)
print(network.weights)
print(network.biases)

print('start gen')
y_test = np.array([network.forward_feed(x) for x in x_test]).reshape((300, 451, 3)) * 255
imageio.imwrite('result.jpg', y_test.astype(np.uint8))
