# -------------------------------------------------------------------
# -------------------------------------------------------------------
## Written by Samantha Jane Alloo (University of Canterbury, New Zealand)
# Contains ideas published in:
# 1) Alloo, Samantha J., et al. "Multimodal intrinsic speckle-tracking (MIST) to extract images of rapidly-varying
# diffuse X-ray dark-field." Scientific Reports 13.1 (2023): 5424.
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
# DEFINING REQUIRED FUNCTIONS
def kspace_alloo(image_shape: tuple, pixel_size: float = 1):
    # Multiply by 2pi for correct values, since DFT has 2pi in exponent
    rows = image_shape[0]
    columns = image_shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size) # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size) # spatial frequencies relating to "columns" in real space
    ky = (2*math.pi*m) # defined by row direction
    kx = (2*math.pi*n) # defined by column direction
    return kx, ky
def lowpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, beyond some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    lowpass_2d = np.exp(-r * (kr ** 2))

    # plt.imshow(lowpass_2d)
    # plt.title('Low-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return lowpass_2d
def highpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a high-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    highpass_2d = 1 - np.exp(-r * (kr ** 2))

    # plt.imshow(highpass_2d)
    # plt.title('High-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return highpass_2d
def midpass_2D(image, r, pixel_size):
    # -------------------------------------------------------------------
    # This function will generate a low-pass filter and suppress the input spatial frequencies, kr of the image, up to some defined
    # spatial frequency r
    # DEFINITIONS
    # image: input image whos spatial frequencies you want to suppress
    # r: spatial frequency you want to suppress beyond [pixel number]
    # pixel_size: physical size of pixel [microns]
    # -------------------------------------------------------------------
    rows = image.shape[0]
    columns = image.shape[1]
    m = np.fft.fftfreq(rows, d=pixel_size)  # spatial frequencies relating to "rows" in real space
    n = np.fft.fftfreq(columns, d=pixel_size)  # spatial frequencies relating to "columns" in real space
    ky = (2 * math.pi * m)  # defined by row direction
    kx = (2 * math.pi * n)  # defined by column direction

    kx2 = kx ** 2
    ky2 = ky ** 2
    kr2 = np.add.outer(ky2, kx2)
    kr = np.sqrt(kr2)

    highpass_2d = 1 - np.exp(-r * (kr ** 2))

    C = np.zeros(columns, dtype=np.complex_)
    C = C + 0 + 1j
    kx, ky = kspace_alloo(image.shape, pixel_size)  # taking x as columns and y as rows
    ikx = kx * C  # (i) * spatial frequencies in x direction (along columns) - as complex numbers ( has "0" in the real components, and "kx" in the complex)
    denom = np.add.outer((-1 * ky), ikx)  # array with ikx - ky (DENOMINATOR)

    midpass_2d = np.divide(complex(1., 0.) * highpass_2d, denom, out=np.zeros_like(complex(1., 0.) * highpass_2d),
                           where=denom != 0)  # Setting output equal to zero where denominator equals zero

    # plt.imshow(np.real(midpass_2d))
    # plt.title('Mid-Pass Filter 2D')
    # plt.colorbar()
    # plt.show()

    return midpass_2d
# -------------------------------------------------------------------
# IMPORT EXPERIMENTAL SPECKLE-BASED PHASE-CONTRAST X-RAY IMAGING DATA

# Experimental parameters
    # gamma: ratio of real and imaginary refractive index coefficients of the sample (some useful ones provided below)
    # wavelength: wavelength of X-ray beam [microns]
    # prop: propagation distance, that ism between the sample and detector [microns]
    # pixel_size: pixel size of the detector [microns]
os.chdir(r'\DATA_WattleFlower_25keV_2m_SBPCXI') # Put the directory where the 'Wattle Flower' data is here
num_masks = 15
gamma = 1403 # Ratio of real to imaginary components of the sample's refractive index
wavelength = 4.959*10**-5 # [microns]
prop = 2*10**6 # [microns]
pixel_size = 9.9 # [microns]

savedir = r'' # Place the directory you want to save the images to here
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
# CALCULATING MULTIMODAL SIGNALS
start = time.time()
coeff_D = []  # Empty lists to store terms required to solve the system of linear equations
coeff_dx = []
coeff_dy = []
lapacaian = []
RHS = []

coefficient_A = np.empty([int((num_masks)), 4, int(rows),
                          int(columns)])  # Empty arrays to put calculated terms in and to perform QR decomposition on
coefficient_b = np.empty([int((num_masks)), 1, int(rows), int(columns)])

for i in range(
        num_masks):  # This forloop will calculate and store all of the requires coefficients for the system of linear equations
    rhs = (1 / prop) * (Ir[i, :, :] - Is[i, :, :])
    lap = Ir[i, :, :]
    deff = (-1) * np.divide(ndimage.laplace(Ir[i, :, :]), pixel_size ** 2)
    dy, dx = np.gradient(Ir[i, :, :], pixel_size)
    dy_r = -2 * dy
    dx_r = -2 * dx

    coeff_D.append(deff)
    coeff_dx.append(dx_r)
    coeff_dy.append(dy_r)
    lapacaian.append(lap)
    RHS.append(rhs)

# Establishing the system of linear equations: Ax = b where x = [Laplacian(1/wavenumber*Phi - D), D, dx, dy]
for n in range(len(coeff_dx)):
    coefficient_A[n, :, :, :] = np.array([lapacaian[n], coeff_D[n], coeff_dx[n], coeff_dy[n]])
    coefficient_b[n, :, :, :] = RHS[n]

identity = np.identity(4)  # This is applying the Tikhonov Regularisation to the QR decomposition
alpha = np.std(
    coefficient_A) / 10000  # This is the optimal Tikhonov regularisation parameter (may need tweaking if the system is overly unstable)
reg = np.multiply(alpha, identity)  # 4x4 matrix representing the Tikhinov regularization on the coefficient array
reg_repeat = np.repeat(reg, rows * columns).reshape(4, 4, rows,
                                                    columns)  # Repeating the regularisation across all pixel positions
zero_repeat = np.zeros(
    (4, 1, rows, columns))  # 4x1 matrix representing the Tikhinov regularization on the righthand-side vector
coefficient_A_reg = np.vstack(
    [coefficient_A, reg_repeat])  # Coefficient matrix of linear system that is Tikhonov regularised
coefficient_b_reg = np.vstack([coefficient_b, zero_repeat])  # RHS of linear system that is Tikhonov regularised


reg_Qr, reg_Rr = np.linalg.qr(coefficient_A_reg.transpose([2, 3, 0, 1]))
# Now here, we just use a solver to solve Rx = Q^tb instead of taking the inverse
reg_x = np.linalg.solve(reg_Rr, np.matmul( np.matrix.transpose(reg_Qr.transpose([2,3,1,0])),coefficient_b_reg.transpose([2,3,0,1])))

lap_phiDF = reg_x[:, :, 0, 0]  # Laplacian term array (Laplacian(1/wavenumber*Phi - D))
DFqr = reg_x[:, :, 1, 0]  # DF array (DF_reg)
dxDF = reg_x[:, :, 2, 0]  # d(DF)/dx array
dyDF = reg_x[:, :, 3, 0]  # d(DF)/dy array

os.chdir(savedir)
DFphiim = Image.fromarray(lap_phiDF).save(
    'LapDFPhiphase_{}.tif'.format(
        'mask' + str(num_masks) + 'e' + str(alpha)))  # Saving solutions of the system of linear equations
DFim = Image.fromarray(DFqr).save('regDFphase_{}.tif'.format('mask' + str(num_masks) + 'e' + str(alpha)))
dxDFim = Image.fromarray(dxDF).save(
    'dxDFphase_{}.tif'.format('mask' + str(num_masks) + 'e' + str(alpha)))
dyDFim = Image.fromarray(dyDF).save(
    'dyDFphase_{}.tif'.format('mask' + str(num_masks) + 'e' + str(alpha)))
print('System of Linear Equations Solved')

# Determing TRUE dark-field signal by filtering the solutions
cutoff = 20  # This can be determined by optimising the SNR and NIQE for a given cut-off parameter value
#rangeOcut = range(0,200,5) # This for-loop can be used to generate 100 filtered dark-field signals for different cut-off parameter values, just place the next 21 lines of code into the for-loop.
#for cutoff in rangeOcut:

i_dyDF = dyDF * (np.zeros((DFqr.shape),
                          dtype=np.complex_) + 0 + 1j)  # (i) * derivative along rows of DF, has "0" in the real components, and "d(DF)/dx" in the complex

insideft = dxDF + i_dyDF
insideftm = np.concatenate((insideft, np.flipud(insideft)),
                           axis=0)  # Mirroring the term inside the Fourier transform to enforce periodic boundary conditions
ft_dx_idy = np.fft.fft2(insideftm)
MP = midpass_2D(ft_dx_idy, cutoff, pixel_size)
MP_deriv = MP * ft_dx_idy  # This is the 'derivative solution' mid-pass filtered

DFqrm = np.concatenate((DFqr, np.flipud(DFqr)), axis=0)
ft_DFqr = np.fft.fft2(DFqrm)
LP = lowpass_2D(ft_DFqr, cutoff, pixel_size)
LP_DFqr = LP * ft_DFqr  # This is the QR derived solution low-pass filtered

combined = LP_DFqr + MP_deriv  # Combining the two solutions (note, two filters sum to 1)

DF_filtered = np.fft.ifft2(combined)  # Inverting Fourier transform to calculate the TRUE dark-field
DF_filtered = np.real(DF_filtered[0:int(rows), :])
DFFim = Image.fromarray(np.real(DF_filtered)).save(
     'DFphase_FilterCombine_{}.tif'.format('mask' + str(num_masks) + 'e' + str(alpha) + 'r' + str(cutoff)))

# Calculating the phase-shifts and attenuation term - could maybe improve by calcualting the Iob for all sets and averaing
ref = Ir[0, :, :]
sam = Is[0, :, :]

lapphi = (ref - sam + prop * np.divide(ndimage.laplace(DF_filtered * ref), pixel_size ** 2)) * (
        (2 * math.pi) / (wavelength * prop * ref))

lapphim = np.concatenate((lapphi, np.flipud(lapphi)), axis=0)
kx, ky = kspace_alloo(lapphim.shape, pixel_size)
kxky = np.add.outer(ky ** 2, kx ** 2)

ft_lapphi = np.fft.fft2(lapphim)
# for i in range(0,10):
e = 1*10**(-1*int(5))  # This regularises the inverse Fourier transform around the origin of Fourier space (may need tweaking)
insideift = ft_lapphi / (kxky + e)
phi = np.fft.ifft2(insideift)
phi = -1 * np.real(phi)  # This is the phase-shifts imparted on the X-ray wave-field by the object
phi = phi[0:int(rows), :]
phi_im = Image.fromarray(phi).save('Phi_e{}.tif'.format(str(e) + 'gamma' + str(gamma) + 'r' + str(cutoff)))
Iob = np.exp(2 * phi / (gamma))  # This is the object's attenuation term
Iob_im = Image.fromarray(Iob).save('Iob_e{}.tif'.format(str(e) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

DF_atten = np.real(DF_filtered / Iob)  # The object's TRUE attenuating-object approximation of the dark-field

DFattim = Image.fromarray(np.real(DF_atten)).save(
'DFatten_{}.tif'.format('mask' + str(num_masks) + 'gamma' + str(gamma) + 'r' + str(cutoff)))

print('Attenuating-object Dark-field Signal Computed')
end = time.time()
total = end - start
print('The compuation time was ' + str(total))