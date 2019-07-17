import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import glob, os, shutil, socket, warnings
import string, gc, time, copy
import datetime
import astropy.units as u
from linetools.lists.linelist import LineList
from linetools.spectra import convolve as lsc
from scipy.special import wofz

try:
    import configparser
except(ImportError):
    import ConfigParser as configparser

warnings.filterwarnings("ignore")

class hires_fitter:

    def __init__(self, specfile, fitrange, fitlines, ncomp, nfill=0, specres=7.0, contval=1.0, coldef=['Wave', 'Flux', 'Err'], Gpriors=None):
        
	""" Class for dealing with MultiNest fitting
        if provided, specfile should be the *full path* to the spectrum
        """
	
       #input information of the galaxy to be fitted
        self.specfile = specfile
        self.fitrange = fitrange
        self.fitlines = fitlines
        self.Gpriors = Gpriors
	self.specres = specres
	self.contval = contval
	self.ncomp = ncomp
        self.nfill = nfill
        if contval<0:
           self.freecont = True
        else:
           self.freecont = False
        
	#constants
        self.small_num = 1e-70
        self.clight  = 2.9979245e5 #km/s
	self.ccgs = 2.9979245e10 #cm/s
        
	#read dataset so it can be used in the fit		
	file = ascii.read(self.specfile)
        obj_wl    = np.asarray(file[coldef[0]], dtype=float)
        obj       = np.asarray(file[coldef[1]], dtype=float)
        obj_noise = np.asarray(file[coldef[2]], dtype=float)

        #Select the spectral range to be used 
        okrange = np.zeros_like(obj_wl, dtype=bool)	
        for i in range(len(self.fitrange)):
           okrange[(obj_wl>self.fitrange[i][0]) & (obj_wl<self.fitrange[i][1])] = True
	
	self.obj = obj[okrange]
	self.obj_noise = obj_noise[okrange]
	self.obj_wl = obj_wl[okrange]
	
	self.velstep = (self.obj_wl[1]-self.obj_wl[0])/self.obj_wl[0]*self.clight
	
	# read in lines from database
	linelist = LineList('Strong', verbose=False)
        
        #Extract line parameters from database, raise error if not found
	self.numlines = len(fitlines)
        linepars = []
        for i in range(self.numlines):
            temp = linelist[self.fitlines[i]]
            if temp is None:
               print 'ERROR: Line {} not found in database. Aborting.'.format(self.fitlines[i])
               return 0
            else:
               linepars.append(temp)
        
        self.linepars = copy.deepcopy(linepars)
        
        #Prepare te mplate to act as filler, we take this to be the same species being fit
        #to avoid chaning the column density limits, but the wave is set to 1000A arbitrarily.
        self.linefill = copy.deepcopy(linepars[0])
        self.linefill['wrest'] = 250 *u.angstrom
        
        #If fitting multiplets the z range spans the spectrum where the first line is expected
	self.zmin = ((self.fitrange[0][0]+0.25)/self.linepars[0]['wrest'].value)-1.
	self.zmax = ((self.fitrange[0][1]-0.25)/self.linepars[0]['wrest'].value)-1.
        
        #For fillers they can be anywhere
        self.zmin_fill = ((np.min(obj_wl)+0.25)/self.linefill['wrest'].value)-1.
	self.zmax_fill = ((np.max(obj_wl)-0.25)/self.linefill['wrest'].value)-1.
        
        #set up parameter limits
        self.cont_lims = np.array((0.9,1.1))
        self.N_lims = np.array((11.5,16))
	self.z_lims = np.array((self.zmin, self.zmax))
        self.b_lims = np.array((1,30))
        self.z_lims_fill = np.array((self.zmin_fill, self.zmax_fill))
     
        #Define start and ending indices for lines of interest
        if self.freecont:
           self.startind = 1
        else:
           self.startind = 0   
        
        self.endind   = self.startind+3*self.ncomp
	
        #Define bounds
        self.bounds = []
        if self.freecont:
          self.bounds.append(self.cont_lims)
	for ii in range(self.ncomp):
	  self.bounds.append(self.N_lims)
	  self.bounds.append(self.z_lims)
	  self.bounds.append(self.b_lims)
        for ii in range(self.nfill):
	  self.bounds.append(self.N_lims)
	  self.bounds.append(self.z_lims_fill)
	  self.bounds.append(self.b_lims)
          
        self.ndim = len(self.bounds)  
	  	  
    def _scale_cube(self, cube):
        
	cube2 = np.copy(cube)
        for ii in range(len(cube)):
            cube2[ii] = cube2[ii]*self.bounds[ii].ptp() + np.min(self.bounds[ii])

        return cube2

    def lnprior(self, p):
        
        ndim = len(p)
        if all(b[0] <= v <= b[1] for v, b in zip(p, self.bounds)):
            
	    pav = 0
	    
	    if self.Gpriors is not None:
	      for par in range(ndim):
	        if self.Gpriors[2*par] != 'none' and self.Gpriors[(2*par)+1] != 'none':
		  val = float(self.Gpriors[2*par])
		  sig = float(self.Gpriors[(2*par)+1])
                  pav  +=  -0.5*(((p[par]-val)/sig)**2 + np.log(2.*np.pi*sig**2))
		  
	    return pav

        return -np.inf
    
    def chi2(self,p):
	
        #reconstruct the spectrum first    
		
	model_spec = self.reconstruct_spec(p)
        
        if np.all(model_spec == 0.):
            return +np.inf, []
        
        ispec2 = 1./((self.obj_noise)**2)    
        
        chi2 = np.nansum(ispec2*(self.obj-model_spec)**2)   
    
        return chi2
         
    def lnlhood(self, p):
        
	#reconstruct the spectrum first    
		
	model_spec = self.reconstruct_spec(p)
        
        if np.all(model_spec == 0.):
            return -np.inf, []

        ispec2 = 1./((self.obj_noise)**2)

        spec_lhood = -0.5*np.nansum((ispec2*(self.obj-model_spec)**2 - np.log(ispec2) + np.log(2.*np.pi)))
        
	#pr = self.lnprior(p)
        
	#if not np.isfinite(pr):
        #    return -np.inf
        
        pr = 0
	
        return spec_lhood + pr, []
    
    def voigt_tau(self, wave, par):
    	""" Find the optical depth at input wavelengths

    	This is a stripped down routine for calculating a tau array for an
    	input line. Built for speed, not utility nor with much error
    	checking.  Use wisely.  And take careful note of the expected
    	units of the inputs (cgs)

    	Parameters
    	----------
    	wave : ndarray
    	  Assumed to be in cm
    	parm : list
    	  Line parameters.  All are input unitless and should be in cgs
    	    par[0] = logN (cm^-2)
    	    par[1] = z
    	    par[2] = b in cm/s
    	    par[3] = wrest in cm
    	    par[4] = f value
    	    par[5] = gamma (s^-1)

    	Returns
    	-------
    	tau : ndarray
    	  Optical depth at input wavelengths
    	"""
    	cold = 10.0**par[0]
    	zp1=par[1]+1.0
    	nujk = self.ccgs / par[3]
    	dnu = par[2]/par[3] 
    	avoigt = par[5]/( 4 * np.pi * dnu)
    	uvoigt = ((self.ccgs / (wave/zp1)) - nujk) / dnu
    	# Voigt
    	cne = 0.014971475 * cold * par[4] 
    	tau = cne * wofz(uvoigt + 1j * avoigt).real / dnu
    	#
    	return tau

    def voigt_model(self, wave, N, b, z, wrest, f, gamma):
        '''Generate a single Voigt model 
	
        input: wave array  :: Assumed in Angstroms; needs to be unitless
        output: absorbed, normalized flux
        '''
	
        tau = self.voigt_tau(wave/1e8, [N,z,b*1e5,wrest/1e8,f,gamma])
        return np.exp(-1*tau)
    	
    def reconstruct_onecomp(self, continuum, N, z, b):
        
        specmodel = np.zeros_like(self.obj)+continuum
	
	for line in range(self.numlines):
           voigt=self.voigt_model(self.obj_wl, N, b, z, self.linepars[line]['wrest'].value, self.linepars[line]['f'], self.linepars[line]['gamma'].value) 
	   specmodel *= voigt
	    
        #return the re-normalized model + emission lines
	if self.specres > self.velstep:
	     specmodel_conv = lsc.convolve_psf(specmodel, self.specres/self.velstep, boundary='wrap')
             return specmodel_conv
	else:
             return specmodel

    def reconstruct_onecomp_fill(self, continuum, N, z, b):
        
        specmodel = np.zeros_like(self.obj)+continuum
	
        voigt=self.voigt_model(self.obj_wl, N, b, z, self.linefill['wrest'].value, self.linefill['f'], self.linefill['gamma'].value) 
	specmodel *= voigt
	    
        #return the re-normalized model + emission lines
	if self.specres > self.velstep:
	     specmodel_conv = lsc.convolve_psf(specmodel, self.specres/self.velstep, boundary='wrap')
             return specmodel_conv
	else:
             return specmodel

    
    def reconstruct_spec(self, p, targonly=False):
        
	#targonly means reconstruct the full profile of the lines without fillers
	
        if self.freecont:
           specmodel = np.zeros_like(self.obj)+p[0]
        else:
           specmodel = np.zeros_like(self.obj)+self.contval
	
	for comp in range(self.ncomp):
	    _N, _z, _b = p[3*comp+self.startind:3*comp+3+self.startind]
	    
	    for line in range(self.numlines):
               voigt=self.voigt_model(self.obj_wl, _N, _b, _z, self.linepars[line]['wrest'].value, self.linepars[line]['f'], self.linepars[line]['gamma'].value) 
	       specmodel *= voigt
	
        if not targonly:
	   for fill in range(self.nfill):
              _N, _z, _b = p[3*fill+self.endind:3*fill+3+self.endind]
             
              voigt=self.voigt_model(self.obj_wl, _N, _b, _z, self.linefill['wrest'].value, self.linefill['f'], self.linefill['gamma'].value) 
	      specmodel *= voigt
            
        #return the re-normalized model + emission lines
	if self.specres > self.velstep:
	     specmodel_conv = lsc.convolve_psf(specmodel, self.specres/self.velstep, boundary='wrap')
             return specmodel_conv
	else:
             return specmodel
    
    def calc_w(self, p, lineid=0):
        
	#Calculate rest frame equivalent width of the 
	#absorption profile
	Wtot = 0

        if self.freecont:
           cont = p[0]
        else:
           cont = self.contval
	
	for comp in range(self.ncomp):
	    _N, _z, _b = p[3*comp+self.startind:3*comp+3+self.startind]
            absorption=(np.zeros_like(self.obj)+cont)*self.voigt_model(self.obj_wl, _N, _b, _z, self.linepars[lineid]['wrest'].value, self.linepars[lineid]['f'], self.linepars[lineid]['gamma'].value) 
	    
	    dlambda = np.diff(self.obj_wl)
	    dlambda = np.insert(dlambda,0,dlambda[0])
	    
	    Wtemp = np.sum( (1-(absorption/cont)) * dlambda )
	    Wtot += Wtemp/(1+_z)
	
	return Wtot
    
    def calc_N(self, p):
        
	#Calculate rest frame equivalent width of the 
	#absorption profile
	Ntot = 0

	allN = p[self.startind::3]
	allz = p[self.startind+1::3]
	
	okN = (allz<10)
	allN = 10**allN[okN]
	
	return np.log10(np.sum(allN))
	
    
    
    def __call__(self, p):
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -np.inf

        lh = self.lnlhood(p)
        if not np.isfinite(lh):
            return -np.inf

        return lh + lp

    def __enter__(self):
        return self
    def __exit__(self, type, value, trace):
        gc.collect()
    
#Routines to analyse the output and give meaningful names to the parameters,
#not in the class,  can be used without instantiating the class
def pc_analyzer(filesbasename, return_sorted=True):

   fstats = filesbasename+'.stats'
   fpostequal = filesbasename+'_equal_weights.txt'
   
   with open(fstats, 'r') as f:
     for ii, line in enumerate(f):
       if line[:6] == 'log(Z)':
         items = line.split()
         lnz = float(items[2])
         lnz_err = float(items[4])
   
   allsamples = np.loadtxt(fpostequal, ndmin=2)
   
   #Remove first column (weights because they are equal) and strip off second one (-2*likelihood)
   
   lhoodsamples   = -0.5*allsamples[:,1] 
   postsamples    = allsamples[:,2:] 
   
   if return_sorted:
      
      postsorted = np.zeros_like(postsamples) 
      if np.max(postsamples[:,0]) < 2:
         startind = 1
      else:
         startind = 0   
      for ii in range(len(postsamples[:,0])):
         if startind>0:
            postsorted[ii,0] = postsamples[ii,0]
         zsort = np.argsort(postsamples[ii,startind+1::3])
         for jj in range(len(zsort)):
                postsorted[ii,3*jj+startind:3*jj+3+startind] = postsamples[ii,3*zsort[jj]+[0,1,2]+startind]
            
      return lnz, lnz_err, lhoodsamples, postsorted
   else:
      return lnz, lnz_err, lhoodsamples, postsamples
   
def get_parnames(ncomp, cont=False):
    
    parameters = []
    if cont:
       parameters.append('Cont')
    for ii in range(ncomp):
       parameters.append('N{}'.format(ii+1))
       parameters.append('z{}'.format(ii+1))
       parameters.append('b{}'.format(ii+1))
    
    return parameters


def readconfig(configfile=None, logger=None):
    
    """Build parameter dictionaries from input configfile.
    """

    booldir = {'True':True, 'False':False}

    input_params = configparser.ConfigParser()
    input_params.read(configfile)

    #Start with mandatory parameters
    if not input_params.has_option('input', 'specfile'):
        raise configparser.NoOptionError("input", "specfile")

    if not input_params.has_option('input', 'wavefit'):
        raise configparser.NoOptionError("input", "wavefit")
    else:
	tmpwavefit  = input_params.get('input', 'wavefit').split(',')
	nwaves = len(tmpwavefit)
	if nwaves % 2 == 1:
	  raise ValueError("Number of wavefit values must be even")
	else:
	  wavefit = []
	  for i in range(nwaves/2):
	    wavefit.append((float(tmpwavefit[2*i]),float(tmpwavefit[2*i+1])))  
        
    if not input_params.has_option('input', 'linelist'):
        raise configparser.NoOptionError("input", "linelist")
    else:
        linelist = input_params.get('input', 'linelist').split(',')
	
    if not input_params.has_option('input', 'coldef'):
        coldef = ['Wave', 'Flux', 'Err']
    else:
        coldef = input_params.get('input', 'coldef').split(',')	
    
    #Paths are desirable but not essential, default to cwd
    if not input_params.has_option('pathing', 'datadir'):
       datadir = './'
    else:
       datadir = input_params.get('pathing', 'datadir')
    
    if not input_params.has_option('pathing', 'outdir'):
       outdir = './'
    else:
       outdir = input_params.get('pathing', 'outdir')

    if not input_params.has_option('pathing', 'chaindir'):
       chaindir = outdir+'fits/'
    else:
       chaindir = outdir+input_params.get('pathing', 'chaindir')

    if not input_params.has_option('pathing', 'plotdir'):
       plotdir = outdir+'plots/'
    else:
       plotdir = outdir+input_params.get('pathing', 'plotdir')
    
    if not input_params.has_option('pathing', 'chainfmt'):
       chainfmt = 'pc_fits_{}_{1}'
    else:
       chainfmt = input_params.get('pathing', 'chainfmt')
    
    #Here deal with Voigt components and fitting of the continuum
    if input_params.has_option('components', 'ncomp'):
       ncomp = np.array(input_params.get('components', 'ncomp').split(','), dtype=int)
    else:
       ncomp = np.array((1,1), dtype=int) 

    if input_params.has_option('components', 'nfill'):
       nfill = int(input_params.get('components', 'nfill'))
    else:
       nfill = 0

    if input_params.has_option('components', 'contval'):
       contval = int(input_params.get('components', 'contval'))
    else:
       contval = 1
    
    #Parameters driving the run
    if input_params.has_option('run', 'dofit'):
       dofit = booldir[input_params.get('run', 'dofit')]
    else:
       dofit = True

    if input_params.has_option('run', 'doplot'):
       doplot = booldir[input_params.get('run', 'doplot')]
    else:
       doplot = True

    run_params = {'specfile': datadir+input_params.get('input', 'specfile'),
                  'wavefit' : wavefit,
                  'linelist': linelist,
		  'coldef'  : coldef,
		  'chaindir': chaindir,
		  'plotdir' : plotdir,
		  'chainfmt': chainfmt,
		  'ncomp'   : ncomp,
		  'nfill'   : nfill,
		  'contval' : contval,
		  'dofit'   : dofit,
		  'doplot'  : doplot}
		  
    if input_params.has_section('pcsettings'):
    
       settingsdict = (dict((opt, booldir[input_params.get('pcsettings',opt)]) 
    		 if input_params.get('pcsettings',opt) in ['True','False'] 
    		 else (opt, input_params.get('pcsettings', opt)) 
    		 for opt in input_params.options('pcsettings')))
    
       run_params['pcsettings'] = settingsdict
		  
    

    return run_params


