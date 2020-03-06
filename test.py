import torch
import matplotlib.pyplot as plt
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



metadata = unpickle("./data2/batches.meta")
metadata = {key.decode('ascii') : value for (key,value) in metadata.items()}
metadata['label_names'] = [value.decode('ascii') for (value) in metadata['label_names']]
classes = metadata['label_names']


batch_1 = unpickle("./data2/data_batch_1")

b1_labels = batch_1[b'labels']

print(batch_1)