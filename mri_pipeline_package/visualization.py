"""
MRI Visualization Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.express as px

# Set the figure size and create subplots for tumor-present MRI images
figsize = [10, 6]
fig = plt.figure()
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Loop through tumor images and display them in a 3x3 grid
for i, ax in enumerate(fig.axes):
    image = cv2.imread(tumor_files[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(image)
    plt.suptitle('MRI with tumor present', fontsize=16)
    plt.subplots_adjust(hspace=0)

# Create a new figure and subplots for tumor-absent MRI images
fig = plt.figure()
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

# Loop through non-tumor images and display them in a 3x3 grid
for i, ax in enumerate(fig.axes):
    image = cv2.imread(nontumor_files[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    ax = plt.subplot(3, 3, i + 1)
    ax.imshow(image)
    plt.suptitle('MRI with no tumor present', fontsize=16)
    plt.subplots_adjust(hspace=0)


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(221)

# Select a random tumor image
i = r.randint(0, len(tumor_files))
im1 = cv2.imread(tumor_files[i], cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([im1], [0], None, [256], [0, 256])

# Plot histogram for MRI with tumor
ax1.hist(im1.ravel(), 256, [0, 256])
ax1.set_title('Histogram for MRI with tumor')

ax2 = fig.add_subplot(222)

# Select a random non-tumor image
j = r.randint(0, len(nontumor_files))
im2 = cv2.imread(nontumor_files[j], cv2.IMREAD_GRAYSCALE)
hist = cv2.calcHist([im2], [0], None, [256], [0, 256])

# Plot histogram for MRI without tumor
ax2.hist(im2.ravel(), 256, [0, 256])
ax2.set_title('Histogram for MRI without tumor')




fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(221)

i= r.randint(0, len(tumor_files))

img = cv2.imread(tumor_files[i],cv2.IMREAD_GRAYSCALE)
#img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
hist_eq = cv2.equalizeHist(img)

hist_eq = cv2.cvtColor(hist_eq,cv2.COLOR_GRAY2RGB)

ax1.imshow(img, cmap= 'gray')
ax1.set_title('Original image')

ax2 = fig.add_subplot(222)

ax2.imshow(hist_eq, cmap= 'gray')
ax2.set_title('Image after histogram equalization')


# Create a 256 x 256 phantom

# Generate and plot a Shepp-Logan phantom image
true_img = shepp_logan(256)
plt.imshow(true_img, cmap='Greys_r')
plt.title("Shepp-Logan Phantom")


# Define Fourier transform functions and visualize
from scipy.fft import fftn, ifftn, fftshift, ifftshift

# Fourier transform in k-space
mri_fft = lambda x: ifftshift(fftn(fftshift(x)))

# Inverse Fourier transform
mri_ifft = lambda x: ifftshift(ifftn(fftshift(x)))

# Apply Fourier transform to the Shepp-Logan phantom
k_space = mri_fft(true_img)

# Visualize the original image and its Fourier transform magnitude
mag = 20 * np.log(np.abs(k_space))
plt.subplot(121), plt.imshow(true_img, cmap='gray')
plt.title("Shepp-Logan Phantom")
plt.subplot(122), plt.imshow(mag, cmap='viridis')
plt.title("Fourier Transform Magnitude")
plt.show()


#Build and apply a Gaussian filter for blurring

# Generate a Gaussian blur mask
ncols, nrows = 256, 256
sigx, sigy = 10, 10
cy, cx = nrows / 2, ncols / 2
x = np.linspace(0, nrows, nrows)
y = np.linspace(0, ncols, ncols)
X, Y = np.meshgrid(x, y)
blurmask = np.exp(-(((X - cx) / sigx) ** 2 + ((Y - cy) / sigy) ** 2))

# Apply the blur mask to the Fourier-transformed image
ftimagep = k_space * blurmask
plt.imshow(np.abs(ftimagep))

# Plot power spectrum before and after blurring
mag_blur = 20 * np.log(np.abs(ftimagep))
plt.subplot(121), plt.imshow(mag, cmap='viridis')
plt.title("Power Spectral Density - Original")
plt.subplot(122), plt.imshow(mag_blur, cmap='viridis')
plt.title("Power Spectral Density - Blurred")
plt.show()





# For reconstructing the phantom from k-space after blurring, ake the inverse transform and show the blurred image
# Reconstruct the image after blurring using inverse Fourier transform
imagep = mri_ifft(ftimagep)
res = Image.fromarray(abs(imagep))
plt.title("Reconstructed Image After Blur")
plt.imshow(np.abs(res))



