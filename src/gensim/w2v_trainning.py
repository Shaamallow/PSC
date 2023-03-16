import re
import numpy as np
import nltk as sl
import os
from matplotlib import pyplot as plt

# Plot a random graph with different colors

def plot_graph(x, y, x_label, y_label, title, color):
    plt.plot(x, y, color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

# generate random colors

def generate_random_color():
    return np.random.rand(3,)

# generate random x and y values

def generate_random_values():
    x = np.random.rand(10,)
    y = np.random.rand(10,)
    return x, y

# generate random n graph on the same plot

color = [generate_random_color() for i in range(10)]

print(color)
