# -------------------------------------------------------------------
# -------------------------------------------------------------------
## Written by Samantha Jane Alloo (University of Canterbury, New Zealand)
# Contains ideas published in the doctoral thesis
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# ------------------------------------------------------------------
# IMPORTING REQUIRED 'MODULES'
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import special
from scipy.fft import fft2, fftshift, ifft2
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.data import astronaut
from skimage.filters import window
from scipy.ndimage import gaussian_filter
from PIL import Image
import os
import polarTransform
import random
import math
# ------------------------------------------------------------------
# ------------------------------------------------------------------
def SuperGauss(amplitude, shift, width, power, x):
    # --------------------------------------------
    # This function enforces periodicty in the polar coordinate image
    # --------------------------------------------
    # DEFINITIONS:
    # amplitude: the height of the Gaussian function [float]
    # shift: the position of the Gaussian function on the x axis [float]
    # with: the width of the Gaussian function (gives the width) [float]
    # power: the power you want your Gaussian function to be raised too. (higher the power, the more 'step like' it is)
    # x: the independent variable you which to evaluate the Gaussian function along [nd np array]
    # --------------------------------------------
    print('The Gaussian Parameters Are:')
    print('Amplitude'+ ' = '+ str(amplitude))
    print('Transverse Shift' + ' = ' + str(shift))
    print('Width' + ' = ' + str(width))
    print('Power' + ' = ' + str(power))
    # --------------------------------------------
    y = amplitude*np.exp(-((x-shift)**2/(width**2))**power)
    normy = y - np.min(y) / (np.max(y) - np.min(y)) # normalised to 1
    return y, normy
# ------------------------------------------------------------------
# ------------------------------------------------------------------
def RectMask(FSimage,height, DistFromCOR,GFstndDev,heightCOR,fromCOR):
    # ------------------------------------------------------------------
    # This function is the rectangular filter that is used to suppress fourier-space frquencies
    # ------------------------------------------------------------------
    # DEFINITIONS:
    # FSimage: the fourier transform of the polar coordinate CT reconstruction (as np array)
    # height: is how many rows, above and below v = 0 line, you want the rectangle to be
    # DistFromCOR: is how many pixels from the COR (in the u direction), you want the rectangles to
    #              start. This is to ensure the main DC component is not cancelled out too
    # FDstndDev: Standard deviation of Gaussian Filter post-applied to the mask if you want to reduce harsh
    #            cut-off frequencies
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    frow = FSimage.shape[0] # Number of rows in Fourier Space Image
    fcolumn = FSimage.shape[1] # Number of columns in Fourier Space Image

    mask = np.ones([int(frow), int(fcolumn)]) # Establishing binary mask with all 1 values

    # Replacing desired rectangles with zeros, this will cancel the spatial frequencies in these regions
    mask[int(frow/2-height):int(frow/2+height), int(fcolumn/2+DistFromCOR):int(fcolumn)] = np.zeros([int(2*height), int(fcolumn/2-DistFromCOR)])
    mask[int(frow/2-height):int(frow/2+height), 0:int(fcolumn/2-DistFromCOR)] = np.zeros([int(2*height), int(fcolumn/2-DistFromCOR)])

    # Anywhere from DistanceFromCOR to COR is now zeroed out with a rectangle of 1 pixel height above and below
    mask[int(frow/2-heightCOR):int(frow/2+heightCOR), int(fcolumn/2-DistFromCOR):int(fcolumn/2-fromCOR)] = np.zeros([2*heightCOR,int(DistFromCOR-fromCOR)])
    mask[int(frow/2-heightCOR):int(frow/2+heightCOR),int(fcolumn/2+fromCOR):int(fcolumn/2+DistFromCOR)] = np.zeros([2*heightCOR,int(DistFromCOR-fromCOR)])

    # Applying a Gaussian Filter to minimise harsh cut-off frequnecy
    mask_GF = gaussian_filter(mask, sigma=int(GFstndDev))
    return(mask_GF, height, DistFromCOR,GFstndDev, heightCOR, fromCOR)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
def CropAboutCOR(image):
    # -------------------------------------------------
    # This function will crop any input CT image (input as double type numpy array) about the centre-of-rotation
    #--------------------------------------------------
    row, col = image.shape
    # --------------------------------------------------
    # Finding first non-zero element traversing along all rows T --> B
    list_firstNonZero = []
    for i in range(0, int(col)):
        for j in range(0, int(row)):
            if ring_CT_uncrop[j, i] != 0:
                #print('Non Zero Element Found')
                nonZero = j
                list_firstNonZero.append(nonZero)
                break
        else:
            print('Element Equals Zero')

    top_nonzero = min(list_firstNonZero) # FIRST NON-ZER0 ROW

    # Finding first non-zero element traversing along all rows on FLIPPED CTa B --> T
    flipCT = np.flipud(ring_CT_uncrop) # Flip in vertical direction

    list_firstNonZero = []
    for i in range(0, int(col)):
        for j in range(0, int(row)):
            if flipCT[j, i] != 0:
                #print('Non Zero Element Found')
                nonZero = j
                list_firstNonZero.append(nonZero)
                break
        else:
            print('Element Equals Zero')

    bot_nonzero = row - min(list_firstNonZero)  # LAST NON-ZER0 ROW

    # Finding first non-zero element traversing along all columns L --> R
    list_firstNonZero = []
    for i in range(0, int(row)):
        for j in range(0, int(col)):
            if ring_CT_uncrop[i, j] != 0:
                #print('Non Zero Element Found')
                nonZero = j
                list_firstNonZero.append(nonZero)
                break
        else:
            print('Element Equals Zero')

    LH_nonzero = min(list_firstNonZero) # FIRST NON-ZER0 COLUMN

    # Finding first non-zero element traversing along all columns on FLIPPED CT (horizontal) R --> L
    flipCT = np.fliplr(ring_CT_uncrop)
    list_firstNonZero = []
    for i in range(0, int(row)):
        for j in range(0, int(col)):
            if flipCT[i, j] != 0:
                #print('Non Zero Element Found')
                nonZero = j
                list_firstNonZero.append(nonZero)
                break
        else:
            print('Element Equals Zero')

    RH_nonzero = col - min(list_firstNonZero)

    print("Rows = " + str(top_nonzero) + ":" + str(bot_nonzero))
    print("Columns = " + str(LH_nonzero) + ":" + str(RH_nonzero))

    image_crop = image[top_nonzero:bot_nonzero,LH_nonzero:RH_nonzero]
    return image_crop
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Reading in the image
directory = r'C:\Users\sal167\Alloo_PhDResearch\Thesis\PythonScripts\RingArtefactSuppression' # directory with CT in it that you want to ring supress - this address can be gotten from file explorer.
imagename = 'slice0250.tif' # CT filename
savedir = r'C:\Users\sal167\Alloo_PhDResearch\Thesis\PythonScripts\RingArtefactSuppression\TEST' # directory you want to save images too

os.chdir(directory) # this command changes the working directory to where your data is, it says, "python please look for the following data here"
ring_CT_uncrop = np.double(np.asarray(Image.open(imagename))) # reads in the tiff image and then converts it to a double precise array
print('Data is read')
ring_CT = CropAboutCOR(ring_CT_uncrop) # this will crop your image exactly about the COR, if your image already is then dont worry this won't do anything
os.chdir(savedir) # this changes the directory to where you want to save all your images
image = Image.fromarray(ring_CT).save('crop{}'.format(str(imagename))) # saves your cropped image

row = ring_CT.shape[0] # shape is a function that extracts the number of dimensions in your numpy array
column = ring_CT.shape[1]
# ------------------------------------------------------------------
# Convert to polar coordinates
ring_polarCT_uncrop, ptSettings = polarTransform.convertToPolarImage(ring_CT, center=[int(row / 2), int(column / 2)]) # converting to polar coordinates with the centre at the centre of the image
# ring_polarCT is the polar coordinate image and ptSettings are the coordinate conversions used for this specific case
image = Image.fromarray(ring_polarCT_uncrop).save('polar{}'.format(str(imagename))) # saving the polar coordinate image
pucrow = ring_polarCT_uncrop.shape[0] # extracting number of rows and columns of this image as we will need to put the zeros in this image back in
puccol = ring_polarCT_uncrop.shape[1]
first_zeropol = np.where(ring_polarCT_uncrop[0,:]==0)[0][0]-1 # finding the first zero value in the rows
ring_polarCT = ring_polarCT_uncrop[:,0:first_zeropol] # cropping out zeros (reduce low freq. artefacts)
prow = ring_polarCT.shape[0] # this is the shape of our cropped image - need to know these values so we can fill it back up appropriately
pcol = ring_polarCT.shape[1]
image = Image.fromarray(ring_polarCT).save('polarcrop{}'.format(str(imagename))) # saving the cropped polar coordinate image
# ------------------------------------------------------------------
# Constructing an appropriate window function using the super-gaussian function, all of these parameters wont need to be changed.
amplitude = 1
shift = pcol/2
width = pcol/2 - 3*shift/100
power = width/10 # one order of magnitude less than the rest
pixels = np.arange(0,ring_polarCT.shape[1]) # this is a 1D array that has all the pixel values, this is required for the super gaussian function

SGy, SGnormy = SuperGauss(amplitude, shift, width, power, pixels) # This generate the 1D supergaussian
fig, ax = plt.subplots()
ax.plot(SGnormy, 'r-', linewidth=2)
ax.set_xlabel("Column Position, [pixels]", fontsize=14)
ax.set_ylabel("Window Filter Magnitude", fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.savefig('WindowFiler_Column.png', dpi=300)
plt.show()
Mask = np.vstack([SGnormy]*int(prow)) # creating a mask equal to the size of the polar coordinate image - makes it 2D rather than the 1D I generated above
image = Image.fromarray(Mask).save('SuperGaussianMask.tif') # saving the mask
ring_polarCT_windowed = Mask*ring_polarCT # applying window function to the polar coordinate
image = Image.fromarray(ring_polarCT_windowed).save('Windowedpolarcrop{}'.format(str(imagename))) # saving the 'windowed' polar coordinate image
fig, ax = plt.subplots() # This plot shows how periodicty is enforced
ax.plot(ring_polarCT[200,:], 'r-', linewidth=2,label = 'Original')
ax.plot(ring_polarCT_windowed[200,:], 'g--', linewidth=2,label = 'Windowed')
ax.set_xlabel("Column Position, [pixels]", fontsize=14)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.legend(fontsize = 12)
plt.savefig('LPComparisonOfWindow_Column.png', dpi=300)
plt.show()
# # ------------------------------------------------------------------
# Taking the fourier transform
fft_Window = np.fft.fftshift(np.fft.fft2(ring_polarCT_windowed)) # apply 2d fourier transform and then shift the origin to the centre of the array
image = Image.fromarray(np.real(np.log(np.abs(fft_Window)**2))).save('realFT{}'.format(str(imagename))) # saving the magnitude - i.e. power spectrum
# ------------------------------------------------------------------
# Doing fourier space filtering: Parameters RectMask(FSimage,height, DistFromCOR,GFstndDev,heightCOR,fromCOR)
mask_GF, height, DistFromCOR,GFstndDev, heightCOR, fromCOR = RectMask(fft_Window,15,7,7,2,1) # generating the rectangular mask we are using to filter (see defined function to know what each of these numbers do)
image = Image.fromarray(mask_GF).save('RectangleMask.tif') # saving the mask
FS_filtered = fft_Window*mask_GF # doing the rectangular filtering
image = Image.fromarray(np.real(FS_filtered)).save('FTFilt{}'.format(str(imagename))) # saving the filtered power spectrum
# ------------------------------------------------------------------
# Taking the inverse fourier transform
Polar_filtered_window = np.fft.ifft2(np.fft.ifftshift(FS_filtered)) # taking the inverse fourier transform to retrieve back the polar coordinate image - note, there are no zeros
# ------------------------------------------------------------------
# Converting back to cartesian coordinates
zeroes = np.zeros([pucrow,int(puccol-pcol)]) # To "undo" the cropping we initially did to polar coordinate image
Polar_filteredFull = np.concatenate((Polar_filtered_window,zeroes),axis=1)
image = Image.fromarray(np.real(Polar_filteredFull)).save('CorrectedPolar{}'.format(str(imagename))) # saving the zero filled polar coorindate image - this is the corrected
Cartesian_filtered = ptSettings.convertToCartesianImage(Polar_filteredFull) # converting back to cartesian coordinate using the settings used to convert to polar
Cartesian_filtered_image = Image.fromarray(np.real(Cartesian_filtered)).save('{}.tif'.format(str(imagename[0:int(len(imagename)-4)])+'distCOR'+str(DistFromCOR)+'Height'+str(height)+'GF'+str(GFstndDev)+'CORHeight'+str(heightCOR)+'fromCOR'+str(fromCOR))) # saving the corrected image with all the parameters in the name


