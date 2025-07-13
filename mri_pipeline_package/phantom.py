"""
Phantom Simulation and Fourier MRI Reconstruction
"""

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

# Install the phantominator package
!pip install phantominator

import matplotlib.pyplot as plt
from phantominator import shepp_logan


