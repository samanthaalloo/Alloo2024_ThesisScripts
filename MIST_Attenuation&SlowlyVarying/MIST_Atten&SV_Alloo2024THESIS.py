# -------------------------------------------------------------------
# -------------------------------------------------------------------
## Written by Samantha Jane Alloo (University of Canterbury, New Zealand)
# Contains ideas published in:
# 1) Alloo, Samantha J., et al. "Dark-field tomography of an attenuating object using intrinsic x-ray speckle
# tracking." Journal of Medical Imaging 9.3 (2022): 031502-031502.
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# IMPORTING THE REQUIRED MODULES
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy
from scipy import ndimage, misc
from PIL import Image
import time
from scipy.ndimage import median_filter, gaussian_filter
import polarTransform
from matplotlib import cm
# -------------------------------------------------------------------
def kspace_alloo(image_shape: tuple, pixel_size: float = 1):
    # Multiply by 2pi for correct values, since DFT has 2pi in exponent
    rows = image_shape[0]
    columns = image_shape[1]
    v = np.fft.fftfreq(rows, d=pixel_size) # spatial frequencies relating to "rows" in real space
    u = np.fft.fftfreq(columns, d=pixel_size) # spatial frequencies relating to "columns" in real space
    return v, u
# -------------------------------------------------------------------
# IMPORT EXPERIMENTAL SPECKLE-BASED PHASE-CONTRAST X-RAY IMAGING DATA

# Experimental parameters
    # gamma: ratio of real and imaginary refractive index coefficients of the sample (some useful ones provided below)
    # wavelength: wavelength of X-ray beam [microns]
    # prop: propagation distance, that ism between the sample and detector [microns]
    # pixel_size: pixel size of the detector [microns]
os.chdir(r'C:\Users\sal167\Alloo_PhDResearch\Thesis\PythonScripts\MIST_Attenuation&SlowlyVarying\DATA_WattleFlower_25keV_2m_SBPCXI') # Put the directory where the 'Wattle Flower' data is here
num_masks = 15
gamma = 1403 # Ratio of real to imaginary components of the sample's refractive index
wavelength = 4.959*10**-5 # [microns]
prop = 2*10**6 # [microns]
pixel_size = 9.9 # [microns]

savedir = r'C:\Users\sal167\Alloo_PhDResearch\Thesis\PythonScripts\MIST_Attenuation&SlowlyVarying\DATA_WattleFlower_25keV_2m_SBPCXI\TEST' # Place the directory you want to save the images to here
# Ensure the reference-speckle and sample-reference-speckle images are
# imported into numpy arrays. An example of such important can be found below
xleft = 29 # Establishing the desired cropping: For the Wattle Flower data this crops out the image headers, which is required
xright = 2529
ytop = 29
ybot = 2129
rows = 2100 # Total number of rows in image (after cropping)
columns = 2500 # Total number of columns in image (after cropping)
Ir = np.empty([int(num_masks),int(rows),int(columns)]) # Establishing empty arrays to put SB-PCXI data into
Is = np.empty([int(num_masks),int(rows),int(columns)])
ff = np.double(np.asarray(Image.open('FF_2m.tif')))[ytop:ybot, xleft:xright] # Flat-field image
for k in range(0,int(num_masks)):
        i = str(k)
        while len(str(i)) < 2:
            i = "0" + i
        # Reading in data: change string for start of filename as required
        dc = np.double(np.asarray(Image.open('DarkCurrent_Y{}.tif'.format(str(i)))))[ytop:ybot, xleft:xright] # Dark-current image
        ir = np.double(np.asarray(Image.open('ReferenceSpeckle_Y{}.tif'.format(str(i)))))[ytop:ybot, xleft:xright] # Reference-speckle image
        isa = np.double(np.asarray(Image.open('SAMPLE_Y{}_T0.tif'.format(str(i)))))[ytop:ybot, xleft:xright] # Sample-reference-speckle image

        ir = (ir - dc)/(ff-dc) # Dark-current and flat-field correcting SB-PCXI images
        isa = (isa - dc)/(ff-dc)

        Is[int(i)] = (isa)
        Ir[int(i)] = (ir)
        print('Completed Reading Data From Mask = ' + str(i))
# -------------------------------------------------------------------

Numerator = []
Denominator = []

# Calculating the Dark-Field signal:
num_masks -= 1 # Calculating terms in the weighted determinant approach
for i in range(num_masks):
    for j in range(i + 1, num_masks + 1):
        print(str(i) + '|' + str(j)) # these are all the pairs that are used to calcaulte different approx of the DDF
        lapir_i = np.divide(scipy.ndimage.laplace(Ir[i]), pixel_size ** 2)
        lapir_iadd1 = np.divide(scipy.ndimage.laplace(Ir[j]), pixel_size ** 2)

        lapir_ioverir_i = np.divide(lapir_i, Ir[i])
        lapir_iadd1overir_iadd1 = np.divide(lapir_iadd1, Ir[j])

        multref = np.multiply(Ir[i], Ir[j])

        samarefb = np.multiply(Is[i], Ir[j])
        sambrefa = np.multiply(Is[j], Ir[i])

        detab = np.subtract(lapir_iadd1overir_iadd1, lapir_ioverir_i)  # the determinant
        mdetab = np.square(np.abs(detab))  # magnitude of determinant

        numab = np.subtract(sambrefa, samarefb)  # numerator
        wnumab = np.divide((np.multiply(detab, numab)), multref)  # weighted numerator

        Numerator.append(wnumab)  # lists who elements are the terms of the numerator for the WD DF
        Denominator.append(mdetab)  # lists who elements are the terms of the denominator for the WD DF
# ---------------------------------------------------------------------------------

NumeratorSum = sum(Numerator) # Summing all of the calculated terms
DenominatorSum = sum(Denominator)
DenominatorSumProp = np.multiply(DenominatorSum, prop)

Num = np.multiply(NumeratorSum, DenominatorSumProp) # Applying a Tikhiniov Regularisation
alpha = np.mean(np.square(DenominatorSumProp)) / 100 # Regularisation parameter (this was found to be the optimal in most cases)
Den = np.add(np.square(DenominatorSumProp), alpha)
Deffreg = np.divide(Num, Den)
os.chdir(savedir)
DeffregIm = Image.fromarray(Deffreg)
DeffregIm = DeffregIm.save('DF_PhaseObjReg_WD{}.tif'.format(str(num_masks + 1) + 'e' + str(alpha)))

# Calculating the Phase-contrast signal:
G2 = Deffreg * prop # Going 'back', this is the function in Alloo et al. named "G2"

G1_WD = [] # In Alloo et al. the function G1 is defined using only 2 mask positions, here wer are using the
           # weighted determinant approach to also calculate the phase contrast signal

for i in range(num_masks):
    for j in range(i + 1, num_masks + 1):
        G1_pair = Is[i] / Ir[i] - (G2 * np.divide(scipy.ndimage.laplace(Ir[j]), pixel_size ** 2)) / Ir[j]
        # the above is one solution for the G1 function using one pair of mask positions
        G1_WD.append(G1_pair)  # contains all G1 functions

# From our calculation of G2, we have the determinant and magnitude of the determinant already calculated: namely the appropraite
# arrays are in the lists detab (for the determinants) and mdetab (for the magnitudes). So now we just weight all of the
# G1 functions and then normalise and we did it! :)

Num_G1WD = []
Den_G1WD = []
for pair in range(0, len(G1_WD)):
    num = mdetab[pair] * G1_WD[pair]  # the weighted numerator
    den = mdetab[pair]  # the normalising denonimator
    Num_G1WD.append(num)
    Den_G1WD.append(den)

Num_G1WDSum = sum(Num_G1WD)
Den_G1WDSum = sum(Den_G1WD)

WDG1 = Num_G1WDSum / Den_G1WDSum  # The weighted determinant function G1

WDG = WDG1 - np.divide(scipy.ndimage.laplace(G2),
                       pixel_size ** 2)  # The weighted determinant function G using the WD G1 and G2


G2Im = Image.fromarray(G2) # Saving all of the calculated functions
G2Im = G2Im.save('G2_AttenObjReg_WD{}.tif'.format(str(num_masks + 1) + 'e' + str(alpha)))

G1Im = Image.fromarray(WDG1)
G1Im = G1Im.save('G1_AttenObjReg_WD{}.tif'.format(str(num_masks + 1) + 'e' + str(alpha)))

GIm = Image.fromarray(WDG)
GIm = GIm.save('G_AttenObjReg_WD{}.tif'.format(str(num_masks + 1) + 'e' + str(alpha)))

mirror_WDG = np.concatenate((WDG, np.flipud(WDG)), axis=0) # mirroring in the vertical direction to enforce
# periodic boundary conditions (may need horizontal too, use np.fliplr)
v, u = kspace_alloo(mirror_WDG.shape, pixel_size)  # Only require one SBXI data set, so just using first in list
numerator = np.fft.fft2(mirror_WDG)
denominator = 1 + math.pi * gamma * prop * wavelength * (np.add.outer(v ** 2, u ** 2))
Iob = np.fft.ifft2(numerator / denominator)
Iob = np.real(Iob)[0:int(rows),0:int(columns)]
Iob_image = Image.fromarray(Iob).save('Iob_Alloo2022_{}.tif'.format(str(gamma)))

# Calculating the attenuating object approximation
Deffatten = Deffreg/Iob
DeffregIm = Image.fromarray(Deffatten)
DeffregIm = DeffregIm.save('DF_AttenObjReg_WD{}.tif'.format(str(num_masks + 1) + 'e' + str(alpha)+'gamma'+str(gamma)))
# ---------------------------------------------------------------------------------