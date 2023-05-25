# **** ****
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

# **** ****
from skimage.feature import canny
from skimage.draw import polygon


# **** ****
sample_image  = np.zeros(   (500, 500),                         # shape
                            dtype=np.double)                    # data type

# **** draw a polygon in the image ****
poly = np.array((   (200, 100),
                    (150, 200),
                    (150, 300),
                    (250, 300),
                    (350, 200)))

# **** ****
rr, cc = polygon(poly[:, 0], poly[:, 1], sample_image.shape)    # rr, cc = row, column

# **** ****
sample_image[rr, cc] = 1


# **** visualize image ****
plt.figure(figsize=(6, 6))
plt.imshow(sample_image, cmap=plt.cm.gray)
plt.title('sample_image')
plt.show()


# **** add gaussian noice to image ****
im = ndi.gaussian_filter(sample_image, 4)
im += 0.2 * np.random.random(im.shape)

# **** use canny to detect edges 
#      large values of sigma indicate that the edge detection method is less sensitive to noise ****
edges1 = canny( im, 
                sigma= 1.0)             # sigma = standard deviation of the Gaussian filter
edges12 = canny(im,
                sigma= 1.3)             # sigma = standard deviation of the Gaussian filter
edges2 = canny( im, 
                sigma= 1.7)             # sigma = standard deviation of the Gaussian filter


# **** display results ****
fig, (ax0, ax1, ax2, ax3) = plt.subplots(   nrows=1,                # number of rows 
                                            ncols=4,                # number of columns
                                            figsize=(12, 4),        # figure size
                                            sharex=True,            # share x axis
                                            sharey=True)            # share y axis

# **** plot original image ****
ax0.imshow(im, cmap='gray')
ax0.axis('off')
ax0.set_title('noisy')


# **** plot canny edge detection ****
ax1.imshow(edges1, cmap='gray')
ax1.axis('off')
ax1.set_title('Canny filter, $\sigma=1.0$')


# **** plot canny edge detection ****
ax2.imshow(edges12, cmap='gray')
ax2.axis('off')
ax2.set_title('Canny filter, $\sigma=1.3$')


# **** plot canny edge detection ****
ax3.imshow(edges2, cmap='gray')
ax3.axis('off')
ax3.set_title('Canny filter, $\sigma=1.7$')


# **** tight layout****
fig.tight_layout()

# **** plot canny edge detection ****
plt.show()