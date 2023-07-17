# Code by : Ian Baeza
#-----Packages/Requiremnts-----
import sys
import warnings
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy import interpolate
from astropy.io import fits
from numpy import savez_compressed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel

print('Welcome to the Isochrone Fitter!\n')

#################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

#----CLASS----

# This code was made thinking in padova isochrones and an adjustment in the HR diagram, in specific for Gaia EDR3 so the first steps should be adjusted to your requirements if this isnt the photometric system that are you using

# Also was thinking using one isochrone or at least 16 (since we have to show relative to other)


class ISO_FIT_HR():
    #_____________________________________________________________________________________________________________________

    #               VARIABLES AND INDICATIONS
    #_____________________________________________________________________________________________________________________

    # The HR diagram will be constructed with the First Band as Magnitude adn the (Second Band - Thrid Band ) as the color

    #_____________________________________________________________________________________________________________________

    #WARNING:       Must have the same lenght and being correlated

    # flux1 : Is an array or list type with the flux of our best selected data from our object and the main Band
    # flux2 : Is an array or list type with the flux of our best selected data from our object this would be the second Band
    # flux3 : Is an array or list type with the flux of our best selected data from our object this would be the third Band
    #
    # flux1err : Is an array or list type with the flux error of our best selected data from our object and the main Band
    # flux2err : Is an array or list type with the flux error of our best selected data from our object this would be the second Band
    # flux3err : Is an array or list type with the flux error of our best selected data from our object this would be the third Band
    #_____________________________________________________________________________________________________________________

    #WARNING:      This was made just to use padova files

    # Isofile    : Is the name of the isochrone(s) (if it's in a diferent directory path+name)
    # Skip_header: Is an Int value wich refers to the lines of header in the file (is required to the correct reading of the file)
    #
    # DEFAULT : 13
    #_____________________________________________________________________________________________________________________

    #WARNING:      Larger numbers implies larger times of calculus it's highly recommend to use a very constrain ranges

    # Modulus_of_distance_range: Is an array or list with the modulus of distance values to try in the fit
    #                            (e.g. np.linspace(14,16,25))

    # Reddening_Range          : Is an array or list with the color values to try in the fit (Av Visual absortion values)
    #                            (e.g. np.linspace(0,2,25))
    #
    # DEFAULT : Modulus_of_distance_range = np.linspace(10,18,50), Reddening_Range = np.linspace(1e-5,2,50),
    #_____________________________________________________________________________________________________________________

    #WARNING:      This values could be find in the output page of padova webpage

    # AyAv1: Rate of absotion of wavelenght observed over visual (used in reddening) asociated with Mag
    # AyAv2: Rate of absotion of wavelenght observed over visual (used in reddening) asociated with first component of color
    # AyAv3: Rate of absotion of wavelenght observed over visual (used in reddening) asociated with secund component of color

    # DEFAULT : AyAv1= 0.83627 (G_mag), AyAv2= 1.08337 (Bp), AyAv3= 0.63439 (Rp)
    #_____________________________________________________________________________________________________________________

    #WARNING:      This values has to be checked previously in the isochrone compared with our data

    # Labels_to_fit: Is an array or list of number which are the labels to refer to stellar stages
    #                that will be used to determine the best fit
    #                (e.g Main secuence, Sub Giant, Red Giant ,AGB )
    #                It's no recomended to use horizontal brach (due mostly of them are not from ssp)

    # Labels_with_issues: Is an array or list of number which are the labels to refer to stellar stages
    #                     that had problems to be determinate as SSP (e.g Horizontal Branch which implies numbers 4,5,6)

    # solution:     Is an string variable with two options ('Huge_spread','Not_considered'), which is used when Labels_with_issues
    #               have in common numbers with Labels_to_fit, this will implie that the code will do a pre-run to assign stellar
    #               stages to the inputed data, the solutions proposed are assign large spread (less statistical weigth) or simply not
    #               consider such stars (special data always will be computed) , the last option use those points that could be asociated
    #               with the same stellar stage in al the variations of the isochrones

    # First_Band : Is an string that is the Label to call the column in the isochrone with the magnitude associated with the first band
    # Second_Band: Is an string that is the Label to call the column in the isochrone with the magnitude associated with the second band
    # Third_Band : Is an string that is the Label to call the column in the isochrone with the magnitude associated with the third band
    #
    # DEFAULT : Values [1,2,3,4,7,8] (Main sequence, Sub Giant, Red Giant, CHEB, E-AGB, TP-AGB)
    # DEFAULT : Values [4,5,6.9] (CHEB, Blue CHEB, Red CHEB, Post AGB, This for Gaia EDR3 isochrone (5,6 are used in young clusters))
    # DEFAULT : 'Huge_spread' (this option is better when the age and metallicity are known, when you try to determine metallicity and/or age is recomended to use 'Not_considered')
    #_____________________________________________________________________________________________________________________

    #WARNING:      Higher Values requires larger amounts of time

    # N_points: Is an int number which refers to an approx number of points that will be used in the isochrone
    #           (to be compared which is nearest with the observed data)

    # DEFAULT : 200
    #_____________________________________________________________________________________________________________________

    # Progress_Bar: Is a bool variable which set if it is showed the progress bar during the computation of best fit

    # DEFAULT : True
    #_____________________________________________________________________________________________________________________

    # SHOW: Is a bool variable which set if it is displayed as interactive plots
    # Save: Is a bool variable which set if it is saved in a pdf the associated plots
    # PDF : Is a string variable , if Save it setted True this will be the name of the pdf

    # DEFAULT : True, False, plots.pdf
    #_____________________________________________________________________________________________________________________

    # initial_guess: Is a bool variable which set if it is given initial values to make the initial estimation of dispersion
    # init_MH:       Is a float variable which set the initial guess of metallicity (the closest one to values given in the isochrone will be selected)
    # init_AGE:      Is a float variable which set the initial guess of AGE         (the closest one to values given in the isochrone will be selected)

    # DEFAULT : False, None, None
    #_____________________________________________________________________________________________________________________

    # INITIAL_TESTING : Is a bool that stand for looking at the initial assigment and if the dispersion it was well estimated
    #                   it is highly recommended to at least run one test before a complete run

    # DEFAULT : True
    #_____________________________________________________________________________________________________________________

    # POWER : Is a bool that will indicate if we can run a very high resolution last run to infer specific values of the stars

    # DEFAULT : 17499
    #_____________________________________________________________________________________________________________________

    # seed : Is a number that will be used as the seed of random number in order to get replicables results

    # DEFAULT : 17499
    #_____________________________________________________________________________________________________________________
    #_____________________________________________________________________________________________________________________
    #_____________________________________________________________________________________________________________________

#https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html # regrading constant of bands in gaia photometry

###########################################################################################################################
    def __init__(self,Isofile,
                      flux1,
                      flux2,
                      flux3,
                      flux1err,
                      flux2err,
                      flux3err,
                      N_points                  = 200,
                      Modulus_of_distance_range = np.linspace(2,18,60),
                      Reddening_Range           = np.linspace(1e-5,2,60),
                      Skip_header               = 13,
                      AyAv1                     = 0.83627,
                      AyAv2                     = 1.08337,
                      AyAv3                     = 0.63439,
                      Labels_to_fit             = [1,2,3,4,7,8],
                      Labels_with_issues        = [4,5,6,9],
                      solution                  = 'Huge_spread',
                      Progress_Bar              = True,
                      SHOW                      = True,
                      Save                      = False,
                      PDF                       = 'best_fit.pdf',
                      First_Band                = 'Gmag',
                      Second_Band               = 'G_BPmag',
                      Third_Band                = 'G_RPmag',
                      initial_guess             = False,
                      init_MH                   = None,
                      init_AGE                  = None,
                      INITIAL_TESTING           = True,
                      POWER                     = False,
                      seed                      = 17499):

        #######################################################################################################
        # Main Variables (required)

        self.INITIAL_TESTING = INITIAL_TESTING  # This will stops after the initial assigment to vizually check if everithing it's ok

        if not INITIAL_TESTING:
            warnings.filterwarnings("ignore")
        #----------
        # Stablish a set of colors in common along the entire code and labels for the Evolutionary Stages
        colors =['cyan','springgreen','olive','lightcoral','cornflowerblue','paleturquoise','grey','fuchsia','gold','saddlebrown']
        self.LB=['Pre-Main Sequence','Main sequence','Sub Giant','RGB','CHEB','CHEB-B','CHEB-R','E-AGB','TP-AGB','post-AGB']
        self.C =ListedColormap(colors, name="my_cmap")
        #----------
        self.POWER=POWER
        self.seed =int(seed)
        np.random.seed(self.seed) # setting a seed in order to get replicables results
        #----------
        # Variables according to CMD 3.7 padova isochrones
        self.Variable1='MH'       # First variable to fit
        self.Variable2='logAge'   # Second Variable to fit
        self.Variable3='int_IMF'  # Used to sort the values
        self.Variable4='label'    # Used to separate the diferent stages

        #----------
        # Varibles according to Gaia DR3 photomotery (aprox generalized values)

        self.G0 = 25.68736690
        self.B0 = 25.33854225
        self.R0 = 24.74789555

        #----------
        #Observed Data

        # Mean Values
        self.y  = -2.5*np.log10(flux1)+self.G0
        self.M2 = -2.5*np.log10(flux2)+self.B0
        self.M3 = -2.5*np.log10(flux3)+self.R0
        self.x  = (self.M2-self.M3).copy()

        #Errors
        self.Mag1e = abs(-2.5*np.log10(flux1+flux1err)+self.G0-self.y)
        self.Mag2e = abs(-2.5*np.log10(flux2+flux2err)+self.B0-self.M2)
        self.Mag3e = abs(-2.5*np.log10(flux3+flux3err)+self.R0-self.M3)
        self.Err_x = self.Mag2e+self.Mag3e

        #self.Mag1eL = -2.5*np.log10(flux1-flux1err)+self.G0
        #self.Mag2eL = -2.5*np.log10(flux2-flux2err)+self.B0
        #self.Mag3eL = -2.5*np.log10(flux3-flux3err)+self.R0

        #Useful Info
        self.width  = self.y.max()-self.y.min()
        self.lenght = self.x.max()-self.x.min()
        self.NDATA  = len(self.y)
        self.none   = np.repeat(False,self.NDATA)

        #Range of increase in dispersion (fixed manually)
        Test = (self.y>(self.y.min()+(self.width*(5./9)))) # Getting proportion of stars over and below 5/9

        ratio = sum(~Test)/sum(Test)
        if ratio<0.5:
            self.Up,self.Low= round(1+ratio,3),round(1-ratio,3)
        else:
            self.Up,self.Low= 1.5,0.5
        print(f'Setting a handicap related to number of sources reliables from:\n {str(self.Up):5>} for the faintest source,\n {str(self.Low):5>} and for the brightest source\n')

        # This values will make that the dispersion of brighter sources will be multiplied by self.Low and the faintest by self.Up
        # putting 1,1 will cancel this effect but it's recomended , since ussually we have much more fainter stars than bright ones so to mitigate this efect we introduce this values

        #-----------------
        #Iso
        self.N_point = N_points
        self.iso      = np.genfromtxt(Isofile,names=True,comments='#',skip_header=Skip_header)
        self.MH       = np.array(list(set(self.iso[self.Variable1])))
        self.AGE      = np.array(list(set(self.iso[self.Variable2])))
        self.MH.sort()
        self.AGE.sort()

        #Stellar Stages
        self.L       = np.array(Labels_to_fit)
        self.L.sort()
        self.L2      = any([False if all(i!=self.L) else True for i in Labels_with_issues]) # To proceed in case of stellar stages with issues
        self.solution=solution
        if self.L2:
            self.L3 =[i for i in Labels_with_issues if i in self.L]
        else:
            self.L3 =[]

        ###
        # keeping just info of our interest
        keep=np.array([True])
        for i in self.L:
            keep= (keep & (self.iso[self.Variable4]!=i))
        self.iso=self.iso[~keep]
        del keep
        ###

        # Variables of ISO
        self.IsoM1    = self.iso[First_Band]
        self.IsoM2    = self.iso[Second_Band]
        self.IsoM3    = self.iso[Third_Band]
        self.iso_BPRP = self.IsoM2-self.IsoM3

        # Useful Values # spliting in an hogogeneus way the dots over the isochrone
        width_ISO     = 0
        lenght_ISO    = 0
        for i in self.L:
            keep = (self.iso[self.Variable4]==i)
            width_ISO  += (self.IsoM1[keep].max()-self.IsoM1[keep].min())
            lenght_ISO += (self.iso_BPRP[keep].max()-self.iso_BPRP[keep].min())

        print('Stellar stage approx % of points: ')
        self.Percentage_of_Point={}
        for i in self.L:
            keep = (self.iso[self.Variable4]==i)
            M = ((self.IsoM1[keep].max()-self.IsoM1[keep].min()))/width_ISO
            C = ((self.iso_BPRP[keep].max()-self.iso_BPRP[keep].min()))/lenght_ISO
            self.Percentage_of_Point[i]=round((M+C)/2.,3)
            print(f'{self.LB[i]:<17}: {str(round(100*self.Percentage_of_Point[i],1)):5^} %')
        print()



        self.HV       = {0:'V',1:'V',2:'H',3:'V',4:'H',5:'V',6:'V',7:'V',8:'H',9:'V'}
        # this dict represent the orientation of the interpolation, V means F(Mag)=Color, H means F(Color)=Mag

        #--------------------
        #Ranges
        self.um    = np.array(Modulus_of_distance_range)
        self.av    = np.array(Reddening_Range)

        #-------------------
        #Absortion
        self.y_abs = float(AyAv1)
        self.m2_abs= float(AyAv2)
        self.m3_abs= float(AyAv3)
        self.x_abs = self.m2_abs-self.m3_abs

        #----------
        # For progress bar
        self.PB     = bool(Progress_Bar)
        self.TOTAL  = len(self.um)*len(self.av)*len(self.MH)*len(self.AGE)
        self.Counter= 0

        #----------
        # Optional Variables
        self.initial_guess= initial_guess
        if initial_guess:
            if type(None)==type(init_MH) or type(None)==type(init_AGE):
                raise Exception('No values were given for init MH or init Age')
            self.init_MH      = self.MH[abs(self.MH-init_MH).argmin()]
            self.init_AGE     = self.AGE[abs(self.AGE-init_AGE).argmin()]

        #----------
        self.SHOW=SHOW
        self.Save=Save
        if self.Save:
            self.PP = PdfPages(PDF)

    #_____________________________________________________________________________________________________________________

    def iso_func(self,Label,MH,AGE,N_dots,Tendency=False): # N_point refers to number of points per section
        # Selection per age and metalicity

        AM       = (self.iso[self.Variable1]==MH)&(self.iso[self.Variable2]==AGE)
        IM_sort  = np.argsort(self.iso[self.Variable3][AM])
        ID        = self.iso[self.Variable4][AM][IM_sort]
        ISO_BPRP  = self.iso_BPRP[AM][IM_sort]
        ISO_Gmag  = self.IsoM1[AM][IM_sort]

        # Isochrone and the interpolation per type and add a complete model

        LX =[]
        LY =[]
        idx=[]
        #--------------------------------------------------------
        for i in range(len(Label)):
            # Selection
            mask_iso = (ID==Label[i])

            if sum(mask_iso)>2:
                N_point=int(N_dots*self.Percentage_of_Point[Label[i]])
                if Tendency:
                    try: # if the next stage has no points this will raise an error
                        idxnp          = list(ID).index(Label[i-len(Label)+1]) # index of the next point (to avoid gaps in isochrone)
                        tendency       = (abs(ISO_BPRP[idxnp-1]-ISO_BPRP[idxnp])<0.15) & (abs(ISO_Gmag[idxnp-1]-ISO_Gmag[idxnp])<0.35)
                                            # If the point follow the tendency of the previus stage we extend the interpolation
                        mask_iso[idxnp]= tendency
                    except:
                        pass # in that case we will just continue

                iso_BPRP = ISO_BPRP[mask_iso]
                iso_Gmag = ISO_Gmag[mask_iso]

                # Orientation (This describes that per stellar stage we can describe the mag in fuction color or vice versa)
                if self.HV[i]=='H':
                    R        =(max(iso_BPRP),min(iso_BPRP))
                    fiso_G = interpolate.interp1d(iso_BPRP,iso_Gmag)
                    x        = np.linspace(R[0],R[1],N_point)
                    y        = fiso_G(x)
                elif self.HV[i]=='V':
                    R        =(max(iso_Gmag),min(iso_Gmag))
                    fiso_BR = interpolate.interp1d(iso_Gmag,iso_BPRP)
                    y        = np.linspace(R[0],R[1],N_point)
                    x        = fiso_BR(y)
                else:
                    print('\n\n\n')
                    raise NameError('The orientation for label 1 wasnt recognise')
                    sys.exit('\n\n\n')
                # Add to model
                LX.extend(list(x))
                LY.extend(list(y))
                idx.extend(list(np.repeat(Label[i],N_point)))
            else: # If we have less than 3 points in a stellar stage we will just add them
                x = ISO_BPRP[mask_iso]
                y = ISO_Gmag[mask_iso]

                LX.extend(list(x))
                LY.extend(list(y))
                idx.extend(list(np.repeat(Label[i],len(x))))
        return np.array(LX),np.array(LY), np.array(idx) # Returns two arrays of the model

    #_____________________________________________________________________________________________________________________
    # Function to create a array of values corresponding to a handicap given certain magnitudes and relative to the whole sample
    def Handicap(self,Y):
        SLOPE = (self.Up-self.Low)/(self.width)
        H     = self.Low+(SLOPE*(Y-self.y.min()))
        return H
    #_____________________________________________________________________________________________________________________

    # the prameter n refers to subsections to estimate from the data dispersion in color, and n_sigma refers to the maximum spread expected due this preliminar assigment compute just the nearest source not the phisicaly relation (diagonal) this would be the correction factor
    def sigmas(self,n=None,unique=True,n_sigma=2,Save=False,color_limit=0.35,DOTS=200,npoints=None):# standar deviation asociated with each source
        if type(npoints)==type(None):
            npoints=25*len(self.L)

        NDATA   = self.NDATA
        n_sigma = int(n_sigma) if 0<n_sigma<6 else 2 # Make sure that it will work
        sDict   = {1:'68.2',2:'95.4',3:'99,7',4:'99.993',5:'99.9999'}

        fig = plt.figure(1,figsize=(12,8))
        gs = gridspec.GridSpec(1,5, height_ratios=[1], width_ratios=[0.8,0.11,0.8,0.07,0.12],left=0.06, right=0.93, bottom=0.08, top=0.95, wspace=0.1, hspace=0.07)

        ax1  = fig.add_subplot(gs[0,0]) # Pre assigment
        ax2  = fig.add_subplot(gs[0,2]) # Gaussian Process (looking for intrinsic noise)
        cbax = fig.add_subplot(gs[0,3])

        # options for solution: "Huge_spread" , "Not_considered"

        x=self.x.copy()
        y=self.y.copy()

        H= self.Handicap(y)

        # we start with common spread and a steper gradient
        s1=np.percentile(y,16)
        s2=np.percentile(y,84)
        self.sy=(H**2)*((s2-s1)/(2))
        s1=np.percentile(x,16)
        s2=np.percentile(x,84)
        self.sx=(H**2)*((s2-s1)/(2))

        #-------------------------
        # Isos
        if self.initial_guess:
            try:
                # We selected the likely isochrone
                LX,LY,ids=self.iso_func(self.L,self.init_MH,self.init_AGE,npoints)
                if sum(LX)<10:
                    raise Exception("The initial guesses constrained the Isochrone to less than 10 points")
            except Exception as e:
                print(f'\n{e}\n')
                sys.exit('The initial Guesses has to be the metalicity and age and has to be cointained in the isochrones provided')
        elif len(self.MH)>4 and len(self.AGE)>4:
            # We selected the 3, extremes isochrones MH and AGE and one middle
            lx1,ly1,ids1=self.iso_func(self.L,self.MH.min(),self.AGE.min(),npoints)
            #lx2,ly2,ids2=self.iso_func(self.L,self.MH.max(),self.AGE.min(),npoints)
            #lx3,ly3,ids3=self.iso_func(self.L,self.MH.min(),self.AGE.max(),npoints)
            lx4,ly4,ids4=self.iso_func(self.L,self.MH.max(),self.AGE.max(),npoints)
            lx5,ly5,ids5=self.iso_func(self.L,self.MH[int(len(self.MH)/2)],self.AGE[int(len(self.AGE)/2)],npoints)

            LX  = list(lx1)+list(lx4)+list(lx5)#+list(lx2)+list(lx3)
            LY  = list(ly1)+list(ly4)+list(ly5)#+list(ly2)+list(ly3)
            ids = np.array(list(ids1)+list(ids4)+list(ids5))#+list(ids2)+list(ids3))
        #-------------------------
        # Light adjustment
        UM=np.linspace(self.um.min(),self.um.max(),30)
        AV=np.linspace(self.av.min(),self.av.max(),30)
        # Progress bar
        num=len(UM)*len(AV)

        # Get an initial fit
        BC,CHI2,W,INDEX=self.chi2_UM_AV(LX,LY,UM,AV,RUN='Pre_Run',TOTAL=num)

        #-----
        self.Counter=0 # reset counter for progress bar
        #-----

        # Get index of best values for distance and reddening
        A,B=W[0][0],W[1][0]

        PreUM=UM[A]
        PreAV=AV[B]

        # Initial stellar stages associated
        IDS=ids[INDEX]
        self.ids_pre_run=IDS.copy()

        # Make a copy of the initial data
        self.original_x=self.x.copy()
        self.original_y=self.y.copy()

        #-------------------------
        # For data asociated with stellar stages with issues
        keep=np.array([True])
        for i in self.L3:
            keep=~(ids==i)&keep
        keep=~keep # these are the stars asociated with stellar stage problem
        self.issues_stars_pre_run=keep.copy()
        #-------------------------

        #Plot
        sc2=ax1.scatter(self.original_x,self.original_y,c=self.ids_pre_run,cmap=self.C,vmin=-0.5,vmax=9.5,alpha=0.8)
        cbar2= plt.colorbar(sc2,cax=cbax,ticks=np.arange(10))
        cbar2.ax.set_yticklabels(self.LB)
        ax1.scatter(LX+self.x_abs*PreAV,LY+PreUM+(self.y_abs*PreAV),c=np.array(ids),cmap=self.C,vmin=-0.5,vmax=9.5,s=60,marker='P',edgecolor='k',lw=2,alpha=0.75,label='Isochrone pre run')
        ax1.set_ylabel('$Magnitude$',fontsize=14)
        ax1.set_xlabel('$Color$',fontsize=14)
        ax1.set_title('Pre Run, stellar assigment',fontsize=14)
        ax1.set_ylim(self.y.max()*1.07,self.y.min()*0.93)
        ax1.set_xlim(min(self.original_x)-0.3,max(self.original_x)+0.3)
        ax1.legend()

        #-------------------------
        # estimating the dispersion per stellar stage
        solution=self.solution # for stellar stages with issues

        #Plot
        #ax2.errorbar(self.x,self.y, yerr=self.Mag1e, xerr=self.Err_x, color='k', ecolor='k', elinewidth=1,lw=0,markersize=5,marker='.',alpha=0.7,zorder=-3)
        ax2.set_ylim(self.y.max()*1.07,self.y.min()*0.93)
        ax2.fill_between([],[],[],alpha=0.5,color='k',label=rf"{sDict[n_sigma]}% confidence interval")
        ax2.plot([],[],color='k',lw=2,alpha=0.9,label='Mean Prediction')


        # Axiliary lists to store the new representative data
        AUX1 = []
        AUX2 = []
        meanx = []
        meany = []

        # Either way we estimate the error four the sources
        DATSX= []
        DATSY= []

        # Kernel used in GP
        kernel = 5**2 * RBF(length_scale=2.0, length_scale_bounds=(0.5,20))     + \
                 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)        + \
                 WhiteKernel(noise_level=0.1**2,noise_level_bounds=(1e-16, 0.5)) #+ \
                 #0.5**2 * RBF(length_scale=1.0)

        print('\nComputing GP for each stellar Stage\n')
        ERRX = interpolate.interp1d(y,self.Err_x+1e-5) # Get approx common errors
        ERRY = interpolate.interp1d(y,self.Mag1e+1e-5) # Get approx common errors

        #-----
        #This is for reorganize the errors obtained for the data
        MASK,Asis=[],np.arange(len(y))
        #----
        for i in self.L:
            # SELECTION AND INTERPOLATION BY STELLAR STAGE
            mask = (self.ids_pre_run==i)
            MASK.extend(list(Asis[mask]))
            NI = sum(mask) #Number of sources associated with this stage
            if NI==0: # In case to note have any star associated with the specific stellar stage
                continue

            print(f'Evolutionary Stage {i} ...')

            X,Y =x[mask],y[mask]

            a,b = X.min(), X.max()
            c,d = Y.min(), Y.max()

            # If our data it's to large we will use a representative sample
            sample=()
            if NI>(6*DOTS*self.Percentage_of_Point[i]):
                sample=[]
                N_sample=int(5.5*DOTS*self.Percentage_of_Point[i])
                rng   = np.random.RandomState(self.seed)
                if   self.HV[i]=='H':# width in X
                    aux  = (b-a)/5
                    aux2 = a
                    var  = X
                elif self.HV[i]=='V':# width in Y
                    aux  = (d-c)/5
                    aux2 = c
                    var  = Y
                for k in range(5):
                    keeP  = (((aux2+(k*aux))<var)&(var<(((k+1)*aux)+aux2)))
                    KN    = sum(keeP)
                    sample.extend(list(rng.choice(np.arange(KN), size=int(self.Handicap(np.mean(var[keeP]))*N_sample*((KN/NI))+1), replace=False)))
                    sample.extend([var.argmin(),var.argmax()])
            #-------print(int(self.Handicap(np.mean(var[keeP]))*N_sample*((KN/NI))+1))
            #------------------------------------
            if i not in self.L3 and NI>10:

                Xtem=X[sample].reshape(-1,1)
                Ytem=Y[sample].reshape(-1,1)

                # Ponit that will be representing our data here by
                X_sampler=np.linspace(a,b,int(DOTS*self.Percentage_of_Point[i])).reshape(-1,1) #-(0.01*(b-a)) #+((b-a)*0.01)
                Y_sampler=np.linspace(c,d,int(DOTS*self.Percentage_of_Point[i])).reshape(-1,1) #-(0.01*(d-c)) #+((d-c)*0.01)

                HH1= self.Handicap(Y_sampler.ravel())
                HH2= self.Handicap(Y)

                #====================
                if self.HV[i]=='H':
                    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=(self.Mag1e[mask][sample])**2, n_restarts_optimizer=2)
                    gaussian_process.fit(Xtem, Ytem)
                    mean_prediction, std_prediction = gaussian_process.predict(X_sampler, return_std=True)

                    std_prediction=std_prediction*HH1

                    # Intrinsic dispersion on the sample in the axis that we not interpolate
                    s1=np.percentile(X,16)
                    s2=np.percentile(X,84)
                    SS=(s2-s1)/(2*(self.lenght))

                    AUX1.extend(list(n_sigma * std_prediction)) # Data new
                    AUX2.extend(list(ERRX(Y_sampler.ravel())+ (n_sigma*SS*HH1)))

                    meanx.extend(list(X_sampler.ravel()))
                    meany.extend(list(mean_prediction))

                    print(gaussian_process.kernel_)

                    # Plot
                    ax2.plot(X_sampler.ravel(),mean_prediction,color=self.C(i/10),lw=2,alpha=0.9,zorder=8)
                    ax2.fill_between(X_sampler.ravel(),
                                     mean_prediction - n_sigma * std_prediction,
                                     mean_prediction + n_sigma * std_prediction,
                                     alpha=0.5,
                                     color=self.C(i/10),zorder=8)
                    # Error for the real data
                    #mean_prediction, std_prediction = gaussian_process.predict(X.reshape(-1,1)+1e-12, return_std=True)  # MSE
                    trend_error = interpolate.interp1d(X_sampler.ravel(),n_sigma* std_prediction) # Get errors
                    std_prediction = trend_error(X)

                    DATSX.extend(list(  self.Err_x[mask]+ (n_sigma*SS*HH2)))     # Data old
                    DATSY.extend(list((std_prediction * HH2)+ self.Mag1e[mask])) # Data old
                #====================
                elif self.HV[i]=='V':
                    gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=(self.Err_x[mask][sample])**2, n_restarts_optimizer=2)
                    gaussian_process.fit(Ytem,Xtem)
                    mean_prediction, std_prediction = gaussian_process.predict(Y_sampler, return_std=True)

                    std_prediction=std_prediction*HH1

                    # Intrinsic dispersion on the sample in the axis that we not interpolate
                    s1=np.percentile(Y,16)
                    s2=np.percentile(Y,84)
                    SS=(s2-s1)/(2*(d-c))

                    AUX2.extend(list(n_sigma * std_prediction))  # Data new
                    AUX1.extend(list(ERRY(Y_sampler.ravel())+(n_sigma *SS*HH1)))

                    meanx.extend(list(mean_prediction))
                    meany.extend(list(Y_sampler.ravel()))

                    print(gaussian_process.kernel_)

                    # Plot
                    ax2.plot(mean_prediction,Y_sampler.ravel(),color=self.C(i/10),lw=2,alpha=0.9,zorder=8)
                    ax2.fill_betweenx(Y_sampler.ravel(),
                                      mean_prediction - n_sigma * std_prediction,
                                      mean_prediction + n_sigma * std_prediction,
                                      alpha=0.5,
                                      color=self.C(i/10),zorder=8)

                    # Error for the real data
                    #mean_prediction, std_prediction = gaussian_process.predict(Y.reshape(-1,1), return_std=True)
                    trend_error = interpolate.interp1d(Y_sampler.ravel(),n_sigma* std_prediction) # Get errors
                    std_prediction = trend_error(Y)

                    DATSX.extend(list((std_prediction * HH2) + self.Err_x[mask]))     # Data old
                    DATSY.extend(list((n_sigma * self.Mag1e[mask])+ (n_sigma*SS*HH2)))# Data old

                #====================
                ax2.errorbar(Xtem,Ytem, yerr=self.Mag1e[mask][sample], xerr=self.Err_x[mask][sample], color='k', ecolor='k', elinewidth=1,lw=0,markersize=5,marker='.',alpha=0.7,zorder=-3)
                print('Done_')
            #------------------------------------
            elif NI>0 and i not in self.L3: # If we dont have enough point in that stage we just let them with higher dispersion
                AUX2.extend(list(n_sigma*(np.mean(self.sx)/(self.lenght)+self.Err_x[mask][sample])))
                AUX1.extend(list(n_sigma*(np.mean(self.sy)/(self.width )+self.Mag1e[mask][sample])))

                meanx.extend(list(X[sample]))
                meany.extend(list(Y[sample]))

                DATSX.extend(list(n_sigma*(np.mean(self.sx)/(2*(self.lenght))+self.Err_x[mask])))
                DATSY.extend(list(n_sigma*(np.mean(self.sy)/(2*(self.width ))+self.Mag1e[mask])))

                ax2.errorbar(X[sample],Y[sample], yerr=self.Mag1e[mask][sample], xerr=self.Err_x[mask][sample], color='k', ecolor='k', elinewidth=1,lw=0,markersize=5,marker='s',alpha=0.7,zorder=-3,label='Few point to make  a GP')

                print('Not enough points , just adding them with high dispersion ... Done')
            #------------------------------------
            elif i in self.L3:
                print(f'it\'s associated with problem, solving through {solution}')

                # For the data we gonna left standar values
                DATSX.extend(list(1.2*(np.std(x)/(b-a)+self.Err_x[mask]))) # Data old
                DATSY.extend(list(1.2*(np.std(y)/(d-c)+self.Mag1e[mask]))) # Data old

                if not any(solution==np.array(['Huge_spread','Not_considered'])):
                    solution='Huge_spread'
                    print('Solution for stellar stages with issues was not properly selected by default is used \"Huge_spread\"')

                elif solution=='Huge_spread':
                    AUX2.extend(list((np.std(self.x)/(b-a) +self.Err_x[mask][sample])))
                    AUX1.extend(list((np.std(self.y)/(d-c) +self.Mag1e[mask][sample])))

                    meanx.extend(list(X[sample]))
                    meany.extend(list(Y[sample]))

                    ax2.errorbar(X[sample],Y[sample], yerr=self.Mag1e[mask][sample], xerr=self.Err_x[mask][sample], color='k', ecolor='b', elinewidth=1,lw=0,markersize=5,marker='D',alpha=0.7,zorder=-3,label='Stellar stage treated with Huge_spread')

                elif solution=='Not_considered':
                    L = list(self.L)
                    L.remove(i)
                    self.L=L
                    self.none[mask]=True

                    ax2.errorbar(X[sample],Y[sample], yerr=self.Mag1e[mask][sample], xerr=self.Err_x[mask][sample], color='k', ecolor='r', elinewidth=1,lw=0,markersize=5,marker='P',alpha=0.7,zorder=-3,label='Stellar stage Not_considered')

                else:
                   sys.exit('Something went wrong in the handling of stellar stages with issues')
                print('Done__')
            #------------------------------------
        ax2.set_xlabel('Color',fontsize=14)
        ax2.set_ylabel('Magnitude',fontsize=14)
        ax2.legend()

        MASK=np.array(MASK).argsort() # to reorganize the errors of the real data

        # Saving the Data
        self.sy=np.array(AUX1)
        self.sx=np.array(AUX2)

        self.GPX=np.array(meanx)
        self.GPY=np.array(meany)

        self.SY =np.array(DATSY)[MASK]
        self.SX =np.array(DATSX)[MASK]

        #print('GPX\t',len(self.GPX))
        #print('GPY\t',len(self.GPY))
        #print('SX\t',len(self.sx))
        #print('SY\t',len(self.sy))

        # Plots of representative Sample and dispersion of data computed
        fig2 = plt.figure(2,figsize=(12,8))
        gs = gridspec.GridSpec(1,3, height_ratios=[1], width_ratios=[0.8,0.11,0.8],left=0.06, right=0.97, bottom=0.08, top=0.95, wspace=0.1, hspace=0.07)

        ax1  = fig2.add_subplot(gs[0,0]) # Pre assigment
        ax2  = fig2.add_subplot(gs[0,2]) # Gaussian Process (looking for intrinsic noise)

        ax1.errorbar(self.GPX,self.GPY, yerr=self.sy, xerr=self.sx, color='k', ecolor='r', elinewidth=1,lw=0,markersize=5,marker='.',alpha=0.7)
        ax1.set_ylim(self.GPY.max()*1.07,self.GPY.min()*0.93)
        ax1.set_xlabel('Color',fontsize=14)
        ax1.set_ylabel('Magnitude',fontsize=14)
        ax1.set_title('Representative Data and their errors')

        ax2.errorbar(self.x,self.y, yerr=self.SY, xerr=self.SX, ecolor='r', elinewidth=1,lw=0,alpha=0.7)
        ax2.scatter(self.x,self.y,s=30,facecolor='none',edgecolor='k',lw=0.75,alpha=0.7,marker='o')
        ax2.set_ylim(self.y.max()*1.07,self.y.min()*0.93)
        ax2.set_xlabel('Color',fontsize=14)
        ax2.set_ylabel('Magnitude',fontsize=14)
        ax2.set_title(' Data and Dispersion computed')


        if self.Save:
            self.PP.savefig(fig,dpi=100) #dpi below 100 substract resolution (due the large amount of plots could be useful)
            self.PP.savefig(fig2,dpi=100)
        if self.SHOW:
            plt.show()

        if self.INITIAL_TESTING:
            if self.Save:
                self.PP.close()
            sys.exit('\n############################################################\n\n\tThe test has ended\n\n############################################################')

        plt.close('all')
        return True



    #_____________________________________________________________________________________________________________________

    def progress(self,count,status='',TOTAL=None,AV=[]): # Progress bar
        total=TOTAL if type(TOTAL)!=type(None) else self.TOTAL
        count=count*len(AV)
        bar_len = 70
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 2)
        bar = '█' * (filled_len) + '-' * (bar_len - filled_len)

        sys.stdout.write('░%s░ %s%s\t ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()
    #_____________________________________________________________________________________________________________________

    def chi2_UM_AV(self,LX,LY,UM,AV,RUN='Searching Best Fit',TOTAL=None,x=None,y=None): # Requires a data set of isochrone
        if (type(x))==type(None):
            y = self.y
            x = self.x

        CHI2=[]
        Asistent=1e12

        # Define std in color for all the cluster through isochrone

        sx=self.sx
        sy=self.sy

        #R=self.y_abs/self.x_abs

        N  = len(x)
        lx = np.tile(LX,(N,1)).transpose()
        ly = np.tile(LY,(N,1)).transpose()

        H=np.arange(N)

        for um in UM:
            umlist = []
            for av in AV:
                G_step  = um+(self.y_abs*av)
                BR_step = self.x_abs*av

                Gfit  = ly+G_step
                BRfit = lx+BR_step

                A = (y-Gfit)
                B = (x-BRfit)

                Distance = np.sqrt(A**2+B**2)
                index    = Distance.argmin(axis=0)

                A=A[index,H]
                B=B[index,H]

                # Reduced Chi2 (less 4 since we are fitting Age,MH,UM,Av)
                Chi2=(np.sum((B/sx)**2)+np.sum((A/sy)**2))/(N-4)

                umlist.append(Chi2)

                if Chi2<Asistent:
                    Asistent = Chi2
                    SINDEX   = np.array(index)

            CHI2.append(umlist)
            if self.PB:
                self.Counter+=1
                self.progress(self.Counter,RUN,TOTAL,AV)
        BestChi=np.nanmin(CHI2)
        return BestChi,np.array(CHI2),np.where(CHI2==BestChi),SINDEX
    # Returns 2D Chi2, best value and index in UM and Av and the index for each source

     #_____________________________________________________________________________________________________________________


    def Differential_reddening_and_UM(self,MH,AGE,UM,AV,INDEX,N_point=None,n_sigmas=3.5,to_far=True,POWER=False):

        if type(N_point)==type(None) and not POWER:
            N_point=3000*len(self.L)
        elif type(N_point)==type(None) and POWER:
            N_point=60000*len(self.L)

        print('\nGetting differential reddening and distance modulus,\t',end='')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

            y = self.y
            x = self.x

            self.L=list(self.L)
            self.L.extend(self.L3)
            # Isochrone with High resolution
            lx,ly,ids = self.iso_func(self.L,MH,AGE,N_point,Tendency=True)

            N  = len(x)
            H  = np.arange(N)

            LX = np.tile(lx,(N,1)).transpose()
            LY = np.tile(ly,(N,1)).transpose()

            G_step  = UM+(self.y_abs*AV)
            BR_step = self.x_abs*AV

            Gfit = LY+G_step
            BRfit = LX+BR_step

            A=(y-Gfit)  # Gap between sources and isochrone
            B=(x-BRfit)

            Distance= np.sqrt(A**2+B**2)

            #----------------
            if POWER: # if we have enough computational resources
                # Minimun ratio allowed between Mag_Gap and Color_Gap
                R=self.y_abs/self.x_abs

                INDEX=Distance.argmin(axis=0)

                # sources could not have less reddening (in terms of visual absortion) than 0 or more than 2
                B=np.ma.masked_array(data=B, mask=(B<-BR_step)).filled(np.nan)
                B=np.ma.masked_array(data=B, mask=(B>2*self.x_abs-BR_step+0.001)).filled(np.nan)

                # min allowed mag gap between dots asuming color gap as reddening
                A=np.ma.masked_array(data=A, mask=(A/B<R-0.001)).filled(np.nan)
                A=np.ma.masked_array(data=A, mask=(abs(A)>1.302)).filled(np.nan) # we add a safety upper limit of 1.302 mag
                # a twenty times brighter/fainter gap from expected is clearly not well asociated
                # even considering the upper limit by binaries not resolved of 0.753
                # Hurley, J. and Tout, C.A.: 1998, Monthly Notices of the Royal Astronomical Society 300, 977. doi:10.1046/j.1365-8711.1998.01981.x.
                # but taking into account the diferential reddening and the diferential distance (at least for the largest clusters could be considerable)
                # we decided to let that value as limit

                Distance= np.sqrt(A**2+B**2)

                #----------------
                # Data where is not possible to explain as ssp through this fit
                HELP=np.isnan(np.nanmin(Distance,axis=0))

                # Fixing distance to been able of using argmin
                Distance=np.ma.masked_array(data=Distance, mask=np.isnan(Distance)).filled(1e6)

                index = Distance.argmin(axis=0)

                A=A[index,H]
                B=B[index,H]

                # also cut sources to far from the tendency expected/estimated
                if to_far:
                    Distance=(A<(n_sigmas*self.SY))&(B<(n_sigmas*self.SX))
                    HELP=~((~HELP)&(Distance))
                    A[HELP]=np.nan
                    B[HELP]=np.nan


                # where source are not clearly part of the stellar stage we keep the initial guess
                index[HELP]=(N_point*INDEX[HELP]/N_point).astype('int')
            else:

                index = Distance.argmin(axis=0)

                A=A[index,H]
                B=B[index,H]

                if to_far:
                    Distance=(A<(n_sigmas*self.SY))&(B<(n_sigmas*self.SX))
                    HELP    = ~Distance
                    A[HELP] = np.nan
                    B[HELP] = np.nan

        Chi2=(np.nansum((B/self.SX)**2)+np.nansum((A/self.SY)**2))/(N-4)

        self.none   = ~(~HELP & ~self.none)  # mask where sources are not well explained through the best fit
        self.Cshift = B        # Differential reddening shift relative to the best fit
        self.Mshift = A        # Differential Distance modulus inferred and relative to the best fit
        self.N_point = N_point # Update the N_point
        print('Computed.')
        return index,Chi2,ids


    #_____________________________________________________________________________________________________________________


    def Complete_fit(self,simple_sigma=False):
        Save=self.Save
        SHOW=self.SHOW
        ########################################
        SIGMAS=self.sigmas(unique=simple_sigma)
        ########################################
        if len(self.MH)==1 and len(self.AGE)==1:
            BestMH  = self.MH[0]
            BestAGE = self.AGE[0]

            X,Y,ids = self.iso_func(self.L,BestMH,BestAGE,self.um,self.av,self.N_point)

            BC,CHI2,W,INDEX=self.chi2_UM_AV(X,Y,x=self.GPX,y=self.GPY)

            savez_compressed('AV-UM_chi2.npz',CHI2=CHI2) # Save the likelyhood

            A,B=W[0][0],W[1][0]

            BestUM=self.um[A]
            BestAV=self.av[B]

            print()
            print('#logAge\tMH\tUm\tAv')
            print(round(BestAGE,5),'\t',round(BestMH,5),'\t',round(BestUM,5),'\t',round(BestAV,5))
            print('\n \chi^2 value')
            print(BC)

            INDEX,Real_CHI,ids=self.Differential_reddening_and_UM(BestMH,BestAGE,BestUM,BestAV,INDEX) # Update the index

            print('\nReal \chi^2 value')
            print(Real_CHI)

            #C = Real_CHI/BC # To make proportionality just multiply this factor by the CHI2 and CHI2_MH_AGE

            PLOT=self.AvUm_graph(CHI2,BestChi,INDEX,ids,UM,AV)


        ########################################
        elif len(self.MH)>4 and len(self.AGE)>4:
            #---
            CHI_MH_AGE=[]
            MHAGE  =[]
            UMAVMAP={}
            num=0
            BC =1e9
            for i in self.MH:
                LMH=[]
                #---
                for j in self.AGE:
                    X,Y,ids= self.iso_func(Label=self.L,MH=i,AGE=j,N_dots=self.N_point)
                    bc,chi2,w,index=self.chi2_UM_AV(X,Y,self.um,self.av,x=self.GPX,y=self.GPY)

                    if bc<BC:
                        BC    = bc
                        CHI2  = chi2
                        W     = w
                        INDEX = index
                        BestMH= i
                        BestAGE=j

                    LMH.append(bc)
                    UMAVMAP[str(num)]=chi2
                    MHAGE.append([i,j])
                    num+=1

                    #print(round(bc,3),'\t',round(BC,3),'\t',round(i,2),'\t',round(j,2),'\t',w,'\t',W)
                #---
                CHI_MH_AGE.append(LMH)
            #---
            CHI2_MH_AGE=np.array(CHI_MH_AGE)

            savez_compressed('MH-AGE-steps.npz',MHAGE=np.array(MHAGE)) # Save likelyhood of MH v/s AGE
            savez_compressed('MH-AGE-AV-UM_chi2.npz',**UMAVMAP)        # Save likelyhood of AV v/s UM per each (MH,AGE) computed

            del MHAGE
            del UMAVMAP

            W2=np.where(CHI2_MH_AGE==BC)

            C,D=W[0][0],W[1][0]

            BestUM = self.um[C]
            BestAV = self.av[D]

            print()
            print('#logAge\tMH\tUm\tAv')
            print(round(BestAGE,5),'\t',round(BestMH,5),'\t',round(BestUM,5),'\t',round(BestAV,5))
            print('\n \chi^2 value')
            print(BC)

            INDEX,Real_CHI,ids=self.Differential_reddening_and_UM(BestMH,BestAGE,BestUM,BestAV,INDEX,POWER=self.POWER) # Update the index

            print('\nReal \chi^2 value')
            print(Real_CHI)

            #C = Real_CHI/BC # To make proportionality just multiply this factor by the CHI2 and CHI2_MH_AGE

            PLOT=self.graph_UM_AV_MH_AGE(CHI2,CHI2_MH_AGE,INDEX,self.N_point,BestUM,BestAV,BestMH,BestAGE,IDS=ids)

        ########################################
        else:
            raise ValueError('The lenght of MH or AGE could not be fit in any of the options')
            sys.exit('Sizes must be larger than (2D) 4x4')

        self.BestAGE= BestAGE
        self.BestMH = BestMH
        self.BestUM = BestUM
        self.BestAV = BestAV
        self.INDEX  = INDEX

        return BestMH,BestAGE,BestUM,BestAV,INDEX
    # Returns The MH, AGE, UM, AV and the index for obs sources with respect to the stellar stages in their best fit


    #_____________________________________________________________________________________________________________________

    # Requires 2D-CHI search, NxN array of CHI AGE and MH, the index of sources matched with isochrone, the number of points of each section and the UM and AV selected, and the MH and AGE selected
    def graph_UM_AV_MH_AGE(self,CHI2_UM_AV,CHI2_MH_AGE,INDEX,N_point,UM,AV,MH,AGE,IDS=None,Save=False):

        BestChi=np.nanmin(CHI2_UM_AV)

        C  = self.C
        LB = self.LB
                #http://stev.oapd.inaf.it/cmd_3.1/faq.html

        fig = plt.figure(1,figsize=(12,8))   # Fig size

        gs = gridspec.GridSpec(3,4, height_ratios=[0.45,0.08,0.45], width_ratios=[0.7,0.05,0.3,0.05],left=0.05, right=0.95, bottom=0.08, top=0.93, wspace=0.05, hspace=0.07)      # Space for each plot and space between them (and the edges)

        ax1 = fig.add_subplot(gs[:,0])   # Data and isochrone
        ax2 = fig.add_subplot(gs[0,2])   # AV UM
        ax3 = fig.add_subplot(gs[2,2])   # MH AGE

        cbax1=plt.subplot(gs[0,3])                  # Place of the color Bar
        cbax2=plt.subplot(gs[2,3])                  # Place of the color Bar


        #---Data Obs---
        ids=IDS[INDEX]
        for i in range(len(self.L)):
            mask = (ids==self.L[i])
            ax1.scatter(self.x[mask],self.y[mask],color=C(self.L[i]/10),label=LB[self.L[i]])

        if any(self.none):
            ax1.scatter(self.x[self.none],self.y[self.none],color='firebrick',marker='+',label='Not well fitted \n as a SSP')

        #----Iso-------
        for i in self.L:
            mask = (self.iso[self.Variable1]==MH)&(self.iso[self.Variable2]==AGE)&(self.iso[self.Variable4]==i)
            ax1.plot(self.iso_BPRP[mask]+self.x_abs*AV, self.IsoM1[mask]+self.y_abs*AV+UM, '-', color='k',lw=2.5,alpha=0.6,zorder=7)


        #------Fig_set-------

        ax1.legend(title='Assumed as:',loc='best')
        ax1.set_ylim(max(self.y)+0.65,min(self.y)-0.65)
        ax1.set_xlim(min(self.x)-0.3,max(self.x)+0.3)
        ax1.set_ylabel('$Magnitude$',fontsize=14)
        ax1.set_xlabel('$Color$',fontsize=14)
        ax1.set_title(r'$\chi^2=$'+str(round(BestChi,3)))

        #-----CHI2_UM_AV-----

        A,B = ( (self.av[1])- (self.av[0]))/2,(self.um[1]-self.um[0])/2
        a,b = min(self.av)-A,max(self.av)+A
        c,d = min(self.um)-B,max(self.um)+B

        IM  = ax2.imshow(CHI2_UM_AV,cmap='gist_stern',extent=(a,b,d,c),aspect='auto')
        cbar= plt.colorbar(IM,cax=cbax1,extend='max')
        ax2.plot([AV]*2,[c,d],color='w',ls=':')
        ax2.plot([a,b],[UM]*2,color='w',ls=':')
        ax2.set_title('Av='+str(round(AV,2))+' and Um='+str(round(UM,2)))
        ax2.set_xlabel(r'$A_v$')
        ax2.set_ylabel('Distance Modulus')
        #ax2.set_aspect(len(self.av)/len(self.um))

        #-----CHI2_MH_AGE-----
        BestChi=np.nanmin(CHI2_MH_AGE)
        WHeRE  =np.where(CHI2_MH_AGE==BestChi)

        a,b = ( (self.AGE[1])- (self.AGE[0]))/2,(self.MH[1]-self.MH[0])/2
        a,b,c,d= (min(self.AGE))-a, (max(self.AGE))+a,min(self.MH)-b,max(self.MH)+b
        extent =a,b,c,d

        A,B=WHeRE[0][0],WHeRE[1][0]


        IM  = ax3.imshow(CHI2_MH_AGE,cmap='nipy_spectral',extent=extent,origin='lower',aspect='auto')
        cbar= plt.colorbar(IM,cax=cbax2,extend='max')
        ax3.plot([AGE]*2,[c,d],color='w',ls=':')
        ax3.plot([a,b],[MH]*2,color='w',ls=':')
        ax3.set_title('MH='+str(round(self.MH[A],2))+' and log(AGE)='+str(round(self.AGE[B],2)))
        #ax3.set_aspect(len(self.AGE)/len(self.MH))
        ax3.set_xlabel('Log(Age) (yr)')
        ax3.set_ylabel('MH (dex)')


        if self.Save:
            self.PP.savefig(fig,dpi=100) #dpi below 100 substract resolution (due the large amount of plots could be useful)

        elif self.SHOW:
            plt.show()

        plt.close()
        return True




#_____________________________________________________________________________________________________________________

    # Requires 2D-CHI search, best value, the index of sources matched with isochrone, the number of points of each section and the UM and AV selected
    def AvUm_graph(self,CHI2,BestChi,INDEX,ids,UM,AV,Save=False):
        a,b = min(self.av),max(self.av)
        c,d = min(self.um),max(self.um)


        C  = self.C
        LB = self.LB
        #http://stev.oapd.inaf.it/cmd_3.1/faq.html

        fig = plt.figure(1,figsize=(12,8))   # Fig size

        gs = gridspec.GridSpec(2,4, height_ratios=[0.5,0.5], width_ratios=[0.65,0.075,0.35,0.05],left=0.1, right=0.95, bottom=0.08, top=0.97, wspace=0.08, hspace=0.2)      # Space for each plot and space between them (and the edges)

        ax1 = fig.add_subplot(gs[:,0])   # Data and isochrone
        ax2 = fig.add_subplot(gs[1,2])   # AV UM
        cbax = fig.add_subplot(gs[1,3])  # colorbar
        #---Data Obs---
        ids=IDS[INDEX]
        for i in range(len(self.L)):
            #mask= (i*N_point<=INDEX)&(INDEX<(i+1)*N_point)
            mask = (ids==self.L[i])
            ax1.scatter(self.x[mask],self.y[mask],color=C(self.L[i]/10),label=LB[self.L[i]])

        if any(self.none):
            ax1.scatter(self.x[self.none],self.y[self.none],color='firebrick',marker='+',label='Not well fitted \n as a SSP')


        ax1.legend(title='Assumed as:',loc='center',bbox_to_anchor=(1.15, 0.55, 0.35, 0.45),markerscale=2,fontsize=15)
        ax1.set_ylim(max(self.y)+0.65,min(self.y)-0.65)
        ax1.set_xlim(min(self.x)-0.3,max(self.x)+0.3)
        ax1.set_ylabel('$Magnitude$',fontsize=14)
        ax1.set_xlabel('$Color$',fontsize=14)
        #----Iso-------
        for i in self.L:
            mask = self.iso['label']==i
            ax1.plot(self.iso_BPRP[mask]+self.x_abs*AV, self.iso_Gmag[mask]+self.y_abs*AV+UM, '-', color='k',lw=2.5,alpha=0.6,zorder=7)

        #-----Chi2-----

        cs2 =ax2.contour(self.av,self.um,CHI2,levels=[BestChi,BestChi*1.5,BestChi*2.5],colors=['lime','cornflowerblue','blue'],linewidths=2,zorder=5)
        IM  = ax2.imshow(CHI2,cmap='gist_stern',extent=(a,b,d,c))
        cbar= fig.colorbar(IM,extend='max',cax=cbax)
        cbar.add_lines(cs2)
        ax2.scatter(AV,UM,s=35,marker='X',color='lime')
        ax2.set_title(r'$\chi^2=$'+str(round(BestChi,3))+' at \nAv='+str(round(AV,2))+' and Um='+str(round(UM,2)),fontsize=7)

        ax2.set_ylabel('$Distance$ $Modulus$ $(Um)$',fontsize=8)
        ax2.set_xlabel('$Visual$ $Absorption$ $(Av)$',fontsize=8)

        #---Fig setup---
        ax1.set_title('$MH='+str(self.MH[0])+'$ ,$Age='+str(round(10**(self.AGE[0]-int(self.AGE[0])),3))+'x10e'+str(int(self.AGE[0]))+'yr$')


        if Save:
            self.PP.savefig(fig,dpi=100) #dpi below 100 substract resolution (due the large amount of plots could be usefull
        elif self.SHOW:
            plt.show()

        plt.close()
        return True

    #_____________________________________________________________________________________________________________________


    #_____________________________________________________________________________________________________________________

    def general_run(self): # General way to use this software
        # Compute the best fit (first will determine std of sources if was setted with stellar stages with issues will make a pre-run)
        #try:
        self.Complete_fit(simple_sigma=False)
        # With the best parameters we infer information from the isochrone
        if self.Save:
            self.PP.close()
        #except:
            #if self.Save:
                #self.PP.close()

            # If you use the Save options always put this after all plots were made to properly save the file
        return True


################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

# Code by Ian Baeza
