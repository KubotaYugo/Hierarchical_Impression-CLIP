import torch
import numpy as np
import matplotlib.pyplot as plt



data = [0, 0.1, 0.25, 0.45, 0.85, 0.999999, 1.0]
bin_width = 0.1

fig, ax = plt.subplots()
bins = np.arange(-1, 1+bin_width, bin_width)
ax.hist(data, bins=bins)
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Frequency')
plt.show()
plt.close()