import tensorflow as tf
import numpy as np

def get_one_hot(targets, depth):
    return np.eye(depth)[np.array(targets).reshape(-1)]

label=get_one_hot(np.repeat(np.arange(10),10),depth=10)
print(label.shape[0])