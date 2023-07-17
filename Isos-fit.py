# Packages
#_______________________________
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from Fitting_ISO_CMD_CC import ISO_FIT_HR as ISOF

#------------------------------
# Info of files

path1= r'/Users/ian/Desktop/Astrostatistics_project/Results/'
path2= r'/Users/ian/Desktop/Astrostatistics_project/'

ISO_light = r'ISO-GEDR3-9-13.7Gyr(0.5)+-2+0.5MH(0.0.25).dat'
ISO_heavy = r'ISO-2.2+0.5_MH(0.135)_10.06-10.16_logAge(0.0055).dat'
ISO_Young = r'Young_Cluster_ISO.dat'

#------------------------------
# Specific Info of Clusters

#############################################
###--Melotte22
GA       = path1+r'Best_Melotte22.fits'
ISO      = path2+ISO_Young
N_point  = 220                         # Number of points in the isochrone per stellar type
Um       = np.linspace(4,8,45)         # Magnitude range
Av       = np.linspace(0.0001, 2, 40)  # Colour range
init_AGE = np.log10(100e6)             # in the same units that the isochrone
init_MH  = 0.05
IniTGuess= True                        # Use initi Guees or default?
LLL      = [0,1,2]                     # Stelar Stages labels from padova Isochrones
SOLUTION = 'Not_considered'            # What we do in stellar stages that we do not are able to reproduce well trough a ssp?

INITIAL_TESTING=False                  # Show Plots and see the dispersion computed? or save it after the whole run


POWER = True                           # Bool that indicate sif we can run a last iteration with very demanding condition
                                       # it's not recomended for personal computers unless your sample is short enough (Open cluster)


#############################################
###--M4
#GA       = path1+r'M4_best.fits'
#ISO      = path2+ ISO_heavy

#N_point  = 300                         # Number of points in the isochrone per stellar type
#Um       = np.linspace(9.5,13.5,45)    # Magnitude range
#Av       = np.linspace(0.0001, 2, 40)  # Colour range
#init_AGE = np.log10(12.02e9)           # in the same units that the isochrone
#init_MH  = -1.65
#IniTGuess= True                        # Use initi Guees or default?
#LLL      = [1,2,3,4,7]                 # Stelar Stages labels from padova Isochrones
#SOLUTION = 'Not_considered'            # What we do in stellar stages that we do not are able to reproduce well trough a ssp?

#INITIAL_TESTING=True                 # Show Plots and see the dispersion computed? or save it after the whole run


#POWER = False                          # Bool that indicate sif we can run a last iteration with very demanding condition
                                       ## it's not recomended for personal computers unless your sample is short enough (Open cluster)


#############################################
###--M22
#GA       = path1+r'M_22-best.fits'
#ISO      = path2+ ISO_heavy

#N_point  = 300                         # Number of points in the isochrone per stellar type
#Um       = np.linspace(11,13,45)       # Magnitude range
#Av       = np.linspace(0.0001, 2, 40)  # Colour range
#init_AGE = np.log10(12e9)              # in the same units that the isochrone
#init_MH  = -1.6
#IniTGuess= True                        # Use init Guees or default?
#LLL      = [1,2,3,4,7]                 # Stelar Stages labels from padova Isochrones
#SOLUTION = 'Huge_spread'               # What we do in stellar stages that we do not are able to reproduce well trough a ssp?

#INITIAL_TESTING=True                   # Show Plots and see the dispersion computed? or save it after the whole run


#POWER = False                          # Bool that indicate sif we can run a last iteration with very demanding condition
                                       ## it's not recomended for personal computers unless your sample is short enough (Open cluster)


#############################################
###--Tuc47
#GA       = path1+r'47-Tuc_best.fits'
#ISO      = path2+ISO_heavy

#N_point  = 300                         # Number of points in the isochrone per stellar type
#Um       = np.linspace(10,14,45)       # Magnitude range
#Av       = np.linspace(0.0001, 2, 40)  # Colour range
#init_AGE = np.log10(13.06e9)           # in the same units that the isochrone
#init_MH  = -0.78
#IniTGuess= True                        # Use initi Guees or default?
#LLL      = [1,2,3,4,7]                 # Stelar Stages labels from padova Isochrones
#SOLUTION = 'Huge_spread'               # What we do in stellar stages that we do not are able to reproduce well trough a ssp?

#INITIAL_TESTING=True                   # Show Plots and see the dispersion computed? or save it after the whole run


#POWER = False                          # Bool that indicate sif we can run a last iteration with very demanding condition
                                       ## it's not recomended for personal computers unless your sample is short enough (Open cluster)

#----Variables secundarias-----

SH = 13 # Skip Header isochrone

# Make to Show when you are testing and saving when you are running
SHOW=INITIAL_TESTING
SAVE=not INITIAL_TESTING


#----Reading Data---

with fits.open(GA) as hdul:
    gc = hdul[1]
    flux1    = gc.data.phot_g_mean_flux
    flux2    = gc.data.phot_bp_mean_flux
    flux3    = gc.data.phot_rp_mean_flux
    flux1err    = gc.data.phot_g_mean_flux_error
    flux2err    = gc.data.phot_bp_mean_flux_error
    flux3err    = gc.data.phot_rp_mean_flux_error

#----------------------------------------------
# Running the Fitter V2.0

# For details of the Variables see the Class
TEST=ISOF(ISO,
          flux1,
          flux2,
          flux3,
          flux1err,
          flux2err,
          flux3err,
          N_points                  = N_point,
          Modulus_of_distance_range = Um,
          Reddening_Range           = Av,
          Skip_header               = SH,
          AyAv1                     = 0.83627,
          AyAv2                     = 1.08337,
          AyAv3                     = 0.63439,
          Labels_to_fit             = LLL,
          Labels_with_issues        = [4,5,6,9],
          solution                  = SOLUTION,
          Progress_Bar              = True,
          SHOW                      = SHOW,
          Save                      = SAVE,
          PDF                       = 'best_fit.pdf',
          First_Band                = 'Gmag',
          Second_Band               = 'G_BPmag',
          Third_Band                = 'G_RPmag',
          initial_guess             = IniTGuess,
          init_MH                   = init_MH,
          init_AGE                  = init_AGE,
          INITIAL_TESTING           = INITIAL_TESTING,
          POWER                     = POWER)

x=TEST.general_run() # Run the code
