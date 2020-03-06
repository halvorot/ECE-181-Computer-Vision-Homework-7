import import_data as data

import matplotlib.pyplot as plt
import numpy as np

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # denormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(data.trainloader)
images, labels = dataiter.next()

# show images
imshow(data.torchvision.utils.make_grid(images))
# print labels
print(' '.join('%s \t' % data.classes[labels[j]] for j in range(data.BATCH_SIZE)))