import matplotlib.pyplot as plt
import numpy as np


def load_lesson(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    header = np.loadtxt(path, delimiter=",", max_rows=1, dtype=str)
    return *data.T, header


def visual_lesson(x, y):
    first_sort = y == 1
    zero_sort = np.logical_not(first_sort)
    plt.scatter(x[zero_sort][:, 0], x[zero_sort][:, 1], label="Zero_sort")
    plt.scatter(x[first_sort][:, 0], x[first_sort][:, 1], label="First_sort")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


def visual_errors(x, y, predict):
    right = (y == predict)
    wrong = np.logical_not(right)
    plt.scatter(x[right][:, 0], x[right][:, 1], color='blue', label="Right")
    plt.scatter(x[wrong][:, 0], x[wrong][:, 1], color='red', label="Wrong")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    file_name = input('Enter file name: ')
    visual_lesson(*load_lesson(file_name))
