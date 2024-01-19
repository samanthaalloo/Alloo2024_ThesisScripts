# -------------------------------------------------------------------
# Written by Samantha Jane Alloo (University of Canterbury, New Zealand)
# Contains ideas published in:
# 1) Alloo, Samantha J., et al. "Tomographic phase and attenuation extraction for a sample composed of unknown
# materials using x-ray propagation-based phase-contrast imaging." Optics Letters 47.8 (2022): 1945-1948.
# 2) Alloo, Samantha J., et al."Recovering phase and attenuation information in an unknown sample using X-ray
# propagation-based phase contrast tomography." AIP Conference Proceedings: Proceedings of the 15th International
# Conference on X-ray Microscopy -- XRM2022 2990, 040002 (2023).
# PLEASE NOTE: If the curve-fit doesn't converge to your data, you need to go into the function and
# adjust the "initial guess" parameters. LINE 65 - 69
# -------------------------------------------------------------------
# Importing required modules
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import curve_fit
import numpy as np
import os
import math
from scipy import ndimage, misc
from lmfit import Model
# -------------------------------------------------------------------
# THE ERROR FUNCTION
def erffunc(x, a, b, c, x_0, l):
    # -------------------------------------------------------------------
    # DEFINITIONS:
    # x: position across line-profile (number of pixels)
    # a: vertical shift of error-function
    # b: amplitude of error-function
    # c: describes height of normalised Gaussian bumps
    # x_0: translational shift (along x axis) of error function
    # l: width of error function
    # -------------------------------------------------------------------
    return a + b * (special.erf((x - x_0) / l)) + c * (x - x_0)/l  * (np.exp(-((x - x_0) ** 2) / (l ** 2)))
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# FUNCTION THAT CALCULATES GAMMA FOR AN INTERFACE
def gammacal(filepath, gamma_guess, sod, odd, wavelength, det_voxel):
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # DEFINITIONS
    # THE LINE PROFILE NEEDS TO BE TAKEN SUCH THAT THERE IS EQUAL DISTANCE EITHER SIDE OF THE INTERFACE
    # filepath: line profile data in CSV format (x and y data needs to be in two separate excel columns, no headers)
    # a=vertical shift, b=amplitude, c='bump' height, o=horizontal shift, and l=width of the erf
    # gammaguess: initial gamma used in phase-retrieval (Paganin type approach) to obtain reconstruction (incorrect)
    # sod: source-to-sample distance [microns]
    # odd: object-to-detector distance [microns]
    # det_voxel: pixel size in detector used for X-ray imaging [microns] (not "pixelsize = det_voxel/M)
    # wavelength: wavelength of X-rays used in imaging [microns]
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    mag = (sod + odd) / sod # magnification
    pix = det_voxel/mag # effective pixel size

    line_profile = np.genfromtxt(filepath, delimiter=',') # reading in CSV line-profile data
    position_pixels = line_profile[:, 0]                  # [number of pixels]
    position_microns = position_pixels * pix              # position the line profile was taken over, in microns

    # if CT reconstruction greyscale values represent the absorption coefficient, use this:
    mu_recon = line_profile[:, 1]                         # [metres^-1]
    mu_recon_um = mu_recon * 10 ** -6                     # absorption coefficient, in microns^-1

    beta_recon = (mu_recon_um*wavelength)/(4*math.pi)     # calculate beta from mu

    # if the CT reconstruction is the imaginary component of the refractive index, beta, use this:
    #beta_recon = line_profile[:, 1]
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Curve fitting the data with erffunc() (established above) using a Levenberg–Marquardt algorithm

    # Utilizing line profile to extract initial guesses for curve fitting
    a_guess = 0.5*((np.sum(beta_recon[0:int(len(beta_recon) / 4)]) / int(len(beta_recon) / 4)) + (np.sum(beta_recon[int(3 * len(beta_recon) / 4):int(len(beta_recon))]) / int(len(beta_recon) / 4)))  # take the first and last quarter of the line profile, find the average of all elements in each portion, take the sum and then halving
    b_guess = ((np.sum(beta_recon[0:int(len(beta_recon)/4)])/int(len(beta_recon)/4)) - (np.sum(beta_recon[int(3*len(beta_recon)/4):int(len(beta_recon))])/int(len(beta_recon)/4))) # take the first and last quarter of the line profile, find the average of all elements in each portion, and then take the difference
    c_guess = a_guess/10 # typically one order of magnitude less than the absolute value of \beta is suffice
    l_guess = np.abs((np.argmin(beta_recon)-np.argmax(beta_recon)))*pix
    x_0_guess = 0.5*np.max(position_microns)
    print(a_guess,b_guess,c_guess,l_guess,x_0_guess)

    p0 = np.array([a_guess,b_guess,c_guess,x_0_guess,l_guess]) # Initial guess array


    fit_coefficients, covariance = curve_fit(erffunc, position_microns, beta_recon, p0=p0, method='lm') # curve-fitting

    error = np.sqrt(np.diag(covariance))                       # compute one standard deviation of errors on parameters
    curvefit_data = erffunc(position_microns, fit_coefficients[0], fit_coefficients[1],
                         fit_coefficients[2], fit_coefficients[3], fit_coefficients[4]) # calculating curve-fitted
                                                                                        # reconstruction values

    print(r'The Curve-Fit Parameters: (Vertical-Shift, Amplitude, Bump-Height, Horizontal-Shift, Width of erf())=')
    print(fit_coefficients)

    print(r'The One-Standard Deviation Errors in the Fit Parameters Are:')
    print(error)

    plt.figure(2, figsize=(15,10))
    plt.title("Line Profile Taken Across Two Materials in a CT Reconstruction", y = 1.05, fontsize=24)
    plt.scatter(position_microns, beta_recon, label='Raw Data')
    plt.plot(position_microns, curvefit_data, 'r-', linewidth=3)
    plt.xlabel("Position (x [\u03BCm] )", fontsize=18)
    plt.ylabel("Reconstructed Attenuation Coefficient (\u03B2(x))", fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    plt.grid('both', 'both')
    plt.show()
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Calculating the True Value of Gamma for the Interface
    C = fit_coefficients[2]
    B = fit_coefficients[1]
    l = fit_coefficients[4]

    tau_guess = (odd * wavelength * gamma_guess) / (mag * 4 * (math.pi)) # calculating the tau parameter, which is related to gamma
    tau_true = tau_guess +  ((math.sqrt(math.pi)) * l ** 2  * C) / (4 * B)
    gamma_true = (tau_true * mag * 4 * (math.pi)) / (odd * wavelength) # calculating the true gamma parameter

    print('The True \u03B3 = \u03B4/\u03B2 for the Given Interface is:')
    print(gamma_true)
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # Calculating the Relative Magnitude of Complex Component of Materials Refractive index
    delta_relative = 2*B*gamma_true
    beta_relative = 2*B

    print('The True Relative \u03B2 for the Given Interface is:')
    print(beta_relative)

    print('The True Relative \u03B4 for the Given Interface is:')
    print(delta_relative)

    # -------------------------------------------------------------------
    # Uncertainity Calculation:
    unc_C = error[2]
    unc_B = error[1]
    unc_l = error[3]

    term_C = ((mag * math.pi ** (3 / 2) * l ** 2) / (odd * wavelength * B)) * unc_C
    term_l = ((2 * mag * math.pi ** (3 / 2) * l * C) / (odd * wavelength * B)) * unc_l
    term_B = ((mag * math.pi ** (3 / 2) * l ** 2 * C) / (odd * wavelength * B ** 2)) * unc_B

    unc_tot = (term_C ** 2 + term_l ** 2 + term_B ** 2) ** (1 / 2)
    print('The Uncertainity in \u03B3 is ' + str(unc_tot))

    unc_delta = ((beta_relative * unc_tot) ** 2 + (gamma_true * unc_B) ** 2) ** (1 / 2)
    print('The Uncertainity in \u03B4 is ' + str(unc_delta))

    print('The Uncertainity in \u03B2 is ' + str(unc_B))

    return(fit_coefficients, error, gamma_true, position_microns, beta_recon, beta_relative, delta_relative)

## TEST DATA RUN: (change directory to whereever you have the line profile stored)
# filepath = r'C:\Users\samja\Documents\PhD Research Work\Error Function Analysis Results\Extension to Lab Data\LP_c.csv'
# gammaguess = 100
# sod = 0.75*10**6 # source-to-object distance, microns
# odd = 1.5*10**6 # object-to-detector distance, microns
# wavelength = 6.199*10**-5 # wavelength, microns
# detector_size = 55 # detector pixel size, microns
# gammacal(filepath, gammaguess, sod, odd, wavelength, detector_size)