import matplotlib
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
matplotlib.rcParams['font.family'] = ['Heiti TC']


def show_tf_image(tensor, title=None):
    if len(tensor.shape) == 4:
        batch_size, h, w, d = tensor.shape
        plt.imshow(tensor[0].reshape(h, w).numpy())
    else:
        h, w, d = tensor.shape
        plt.imshow(tensor.reshape(h, w).numpy())

    if title:
        plt.title(title)
    plt.show()
