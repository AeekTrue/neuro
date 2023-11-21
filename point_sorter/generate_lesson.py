import numpy as np
import time

num_examples = 1000
num_inputs = 2
prefix = 'circle'  # round(time.time())

training_file_name = f'training_data_{prefix}.csv'
test_file_name = f'test_data_{prefix}.csv'
head = ', '.join(['in' + str(x) for x in range(1, num_inputs + 1)])


def sort_func(x, y) -> np.ndarray:
    return (x - 0.5)**2 + (y - 0.5)**2 < 0.1


training_x = np.random.random((num_inputs, num_examples))
print(training_x[:, :10])

training_y = sort_func(*training_x)
print(training_y[:10])


np.savetxt(training_file_name, np.concatenate((training_x, np.atleast_2d(training_y)), 0).T, header=head+', out', delimiter=',')

test_x = np.random.random((num_inputs, num_examples))
np.savetxt(test_file_name, test_x.T, header=head, delimiter=',')
