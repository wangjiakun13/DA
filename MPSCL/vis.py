import matplotlib.pyplot as plt
import numpy as np
import os


path = 'e:/DA/MPSCL/ct_train_slice553.npy'

data = np.load(path)
#show data before normalization
data = data * 255
data = data.astype(np.uint8)
plt.imshow(data)
plt.show()