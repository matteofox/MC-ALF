import numpy as np
import astropy.io.fits as fits
import astropy.io.ascii as ascii
import glob, os, shutil, socket, warnings
import string, gc, time, copy
import datetime
import astropy.units as u
from astropy.convolution import convolve, convolve_fft, Gaussian1DKernel
from scipy.signal import convolve as scipy_convolve
from linetools.lists.linelist import LineList
from scipy.special import wofz
from astropy.stats import sigma_clipped_stats

try:
    import configparser
except(ImportError):
    import ConfigParser as configparser

warnings.filterwarnings("ignore")

class als_fitter:

    def __init__(self, specfile, fitrange, fitlines, ncomp, nfill=0, specres=[7.0], contval=[1.0], Nrange=[11.5,16], \
                 brange=[1,30], zrange=None, Nrangefill=[11.5,16], brangefill=[1,30], wrangefill=None, coldef=['Wave', 'Flux', 'Err'], \
                 Gpriors=None, Asymmlike=False, oversample=1,debug=False):
        
        """ Class for dealing with MultiNest fitting
        if provided, specfile should be the *full path* to the spectrum
        """
                
       #input information of the galaxy to be fitted
        self.debug = debug
        self.specfile = specfile
        self.fitrange = fitrange
        self.fitlines = fitlines
        self.Gpriors = Gpriors
        self.Asymmlike = Asymmlike
        self.oversample = int(oversample)
        if self.Asymmlike:
         print("Running asymmetric likelihood")
        self.specres = specres
        self.contval = contval
        self.ncompmin = ncomp[0]
        self.ncompmax = ncomp[1]
        self.nfill = nfill
        if len(contval)>1:
           self.freecont = True
        else:
           self.freecont = False
        
        if len(specres)>1:
           self.freespecres = True
        else:
           self.freespecres = False

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
        self.numfitranges = len(self.fitrange)
        for i in range(self.numfitranges):
           okrange[(obj_wl>self.fitrange[i][0]) & (obj_wl<self.fitrange[i][1])] = True
        
        self.obj = obj[okrange]
        self.obj_noise = obj_noise[okrange]
        self.obj_wl = obj_wl[okrange]
        
        velsteps = (self.obj_wl[1:]-self.obj_wl[:-1])/self.obj_wl[1:]*self.clight
        mn, med, stddev = sigma_clipped_stats(velsteps)
        
        #Define typical velocity step of array
        self.velstep = med
        #Find if spectrum has jumps, need a +1 for the index because velsteps is shorter than wave array
        #then add last element to the indices
        self.jumpind = (np.where(velsteps>med+5*stddev)[0])
        self.jumpind = np.append(self.jumpind, len(self.obj_wl)-1)
        
        
        #Now work out an oversampled wave grid 
        if self.oversample>1:
           
           startstep = self.obj_wl[1]-self.obj_wl[0]
           endstep   = self.obj_wl[self.jumpind[0]]-self.obj_wl[self.jumpind[0]-1]
           self.obj_wl_over = np.linspace(self.obj_wl[0]-startstep/self.oversample, self.obj_wl[self.jumpind[0]]+endstep/self.oversample, num=self.oversample*(1+self.jumpind[0]))
        
           if len(self.jumpind)>1:
              for ii in range(len(self.jumpind)-1):
                   startstep = self.obj_wl[self.jumpind[ii]+2]-self.obj_wl[self.jumpind[ii]+1]
                   endstep   = self.obj_wl[self.jumpind[ii+1]]-self.obj_wl[self.jumpind[ii+1]-1]
                   self.obj_wl_over = np.concatenate((self.obj_wl_over, np.linspace(self.obj_wl[self.jumpind[ii]+1]-startstep/self.oversample, self.obj_wl[self.jumpind[ii+1]]+endstep/self.oversample, num=self.oversample*(self.jumpind[ii+1]-self.jumpind[ii]))))
        else:
           self.obj_wl_over = np.copy(self.obj_wl)  
        
        # read in lines from database
        linelist = LineList('ISM', verbose=False)       
        
        #Extract line parameters from database, raise error if not found
        self.numlines = len(fitlines)
        linepars = []
        for i in range(self.numlines):
            temp = linelist[self.fitlines[i]]
            if temp is None:
               print('ERROR: Line {} not found in database. Aborting.'.format(self.fitlines[i]))
               return 0
            else:
               #Update values for CrII from atomic database in RCooke ALIS code
               if self.fitlines[i] == 'CrII 2066':
                  temp['f'] = 0.0512
                  temp['gamma'] = 4.17E8 /u.s
               if self.fitlines[i] == 'CrII 2062':
                  temp['f'] = 0.0759
                  temp['gamma'] = 4.06E8 /u.s
               if self.fitlines[i] == 'CrII 2056':
                  temp['f'] = 0.103
                  temp['gamma'] = 4.07E8 /u.s
               
               #Finally store
               linepars.append(temp)
   
        
        self.linepars = copy.deepcopy(linepars)
        
        #Prepare the template to act as filler, we take this to be the same species being fit
        #to avoid chaning the column density limits, but the wave is set to 1000A arbitrarily.
        self.linefill = copy.deepcopy(linepars[0])
        self.linefill['wrest'] = 250 *u.angstrom
        
        #set up parameter limits
        self.cont_lims = np.array(contval)
        self.res_lims = np.array(specres)
        
        self.N_lims = np.array(Nrange)
        self.N_lims_fill = np.array(Nrangefill)
        
        self.b_lims = np.array(brange)
        self.b_lims_fill = np.array(brangefill)
        
        #For fillers they can be anywhere unless wrangefill is set
        self.z_lims = []
        for zz in range(self.ncompmax):
          if zrange is None:
            #If fitting multiplets the z range spans the spectrum where the first line is expected
            zmin = ((self.fitrange[0][0]+0.25)/self.linepars[0]['wrest'].value)-1.
            zmax = ((self.fitrange[0][1]-0.25)/self.linepars[0]['wrest'].value)-1.
          elif len(zrange) == 2:
            zmin = zrange[0]
            zmax = zrange[1]
          elif len(zrange) >= 2*self.ncompmax:
            zmin = zrange[2*zz+0]
            zmax = zrange[2*zz+1]
          else:
            print('Zrange keyword not understood. Aborting.')
            return 0
          self.z_lims.append(np.array((zmin, zmax)))
        
       
        #if zrange is not None:
        #  self.z_lims = zrange
        #else:
        #  #If fitting multiplets the z range spans the spectrum where the first line is expected
        #  self.zmin = ((self.fitrange[0][0]+0.25)/self.linepars[0]['wrest'].value)-1.
        #  self.zmax = ((self.fitrange[0][1]-0.25)/self.linepars[0]['wrest'].value)-1.
        #  self.z_lims = np.array((self.zmin, self.zmax))
        
        #For fillers they can be anywhere unless wrangefill is set
        self.z_lims_fill = []
        for zz in range(self.nfill):
          if wrangefill is None:
            zmin_fill = ((np.min(self.obj_wl)+0.25)/self.linefill['wrest'].value)-1.
            zmax_fill = ((np.max(self.obj_wl)-0.25)/self.linefill['wrest'].value)-1.
          elif len(wrangefill) == 2:
            zmin_fill = (wrangefill[0]/self.linefill['wrest'].value)-1.
            zmax_fill = (wrangefill[1]/self.linefill['wrest'].value)-1.
          elif len(wrangefill) == 2*self.nfill:
            zmin_fill = (wrangefill[2*zz+0]/self.linefill['wrest'].value)-1.
            zmax_fill = (wrangefill[2*zz+1]/self.linefill['wrest'].value)-1.
          else:
            print('Wrangefill keyword not understood. Aborting.')
            return 0
          self.z_lims_fill.append(np.array((zmin_fill, zmax_fill)))

        #Define start and ending indices for lines of interest (first space is for ncomp)
        self.startind = 0
        
        if self.freecont:
           self.startind += 1
        if self.freespecres:
           self.startind += 1
                
        self.endind   = self.startind+3*self.ncompmax+1
        
        #Find where the data is detected at 10 sigma
        gauss = np.random.normal(size=len(self.obj))
        self.gauss_cdf = [(gauss>3).sum(), (gauss>4).sum(), (gauss>5).sum()]
        self.gracenum = 0.01*len(self.obj)
        
        #Define bounds
        self.bounds = []
        if self.freespecres:
          self.bounds.append(self.res_lims)
        if self.freecont:
          self.bounds.append(self.cont_lims)
          
        self.bounds.append(ncomp) #This is a tuple
        for ii in range(self.ncompmax):
          self.bounds.append(self.N_lims)
          self.bounds.append(self.z_lims[ii])
          self.bounds.append(self.b_lims)
        for ii in range(self.nfill):
          self.bounds.append(self.N_lims_fill)
          self.bounds.append(self.z_lims_fill[ii])
          self.bounds.append(self.b_lims_fill)
        
        self.ndim = len(self.bounds)  
                          
    def _scale_cube_pc(self, cube):
        
        cube2 = np.copy(cube)
        for ii in range(len(cube)):
            cube2[ii] = cube2[ii]*self.bounds[ii].ptp() + np.min(self.bounds[ii])
            if ii == self.startind:
               cube2[ii] = int(cube2[ii])

        return cube2

    def _scale_cube_mn(self, cube, ndim, nparam):
        
        for ii in range(ndim):
            cube[ii] = cube[ii]*self.bounds[ii].ptp() + np.min(self.bounds[ii])
        
        return cube
        
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
         
    def lnlhood_pc(self, p):
        
        zind = 1+(1+self.startind)+np.arange(self.ncompmax)*3
                
        #if self.ncompmax<10:
        #if not all(p[zind] == np.sort(p[zind])):
        #    return -np.inf, []
        
        lhood =  self.lnlhood_worker(p)
        if self.debug:
           print(lhood)
        
        return lhood, []
    
    def lnlhood_dy(self, p):
        
        zind = 1+(1+self.startind)+np.arange(self.ncompmax)*3
        
        #if self.ncompmax<10:
        #  if not all(p[zind] == np.sort(p[zind])):
        #    return -np.inf
        
        return  self.lnlhood_worker(p)
   
    def lnlhood_mn(self, p, ndim, nparam):
        
        zind = 1+(1+self.startind)+np.arange(self.ncompmax)*3
        parr = np.array([p[x] for x in range(ndim)])
        
        #if self.ncompmax<10:
        #  if not all(parr[zind] == np.sort(parr[zind])):
        #    return -np.inf
        
        lhood =  self.lnlhood_worker(p)
        
        return lhood

    def lnlhood_worker(self, p):

        #reconstruct the spectrum first    
        model_spec = self.reconstruct_spec(p)
        
        ispec2 = 1./((self.obj_noise)**2)

        spec_lhood = -0.5*np.nansum((ispec2*(self.obj-model_spec)**2 - np.log(ispec2) + np.log(2.*np.pi)))
        
        if self.Asymmlike:
          
          resid = (self.obj-model_spec)/self.obj_noise
          
          if (resid>5).sum() > self.gauss_cdf[2]+self.gracenum:
             return -np.inf
          elif (resid>4).sum() > self.gauss_cdf[1]+self.gracenum:
             return -np.inf
          
          #import matplotlib.pyplot as mp
          #mp.plot(self.obj)
          #mp.plot(model_spec)
          #mp.show()
          
          #if (resid>4).sum() > 0.001*self.fitchan:
          #   return -np.inf, []
          
          #mp.hist(resid_2sig, bins=75, range=(2,25), histtype='step')
          #mp.hist(gauss_2sig, bins=75, range=(2,25), histtype='step', lw=2)
          
          #mp.show()
          
          #badres = (model_spec<self.obj)
          #if badres.sum()>0:
          #   spec_lprio = -np.inf #1e10*(-0.5*np.nansum((ispec2[badres]*(self.obj[badres]-model_spec[badres])**2 - np.log(ispec2[badres]) + np.log(2.*np.pi))))
          #else:
          #  spec_lprio = 0
             
          #return spec_lhood+spec_lprio, []   
        
        return spec_lhood 

    
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
        
    def reconstruct_onecomp(self, specresolution, continuum, N, z, b):
        
        specmodel = np.ones_like(self.obj_wl_over)
        
        for line in range(self.numlines):
           voigt=self.voigt_model(self.obj_wl_over, N, b, z, self.linepars[line]['wrest'].value, self.linepars[line]['f'], self.linepars[line]['gamma'].value) 
           specmodel *= voigt
            
        #return the re-normalized model normalized by continuum
        if specresolution > self.velstep:
             specmodel = self.convolve_model(specmodel, specresolution)
        
        #Now rebin to match original wl array and return
        specmodel *= continuum
        
        return np.mean(specmodel.reshape(len(self.obj),self.oversample), axis=1)

    def reconstruct_onecomp_fill(self, specresolution, continuum, N, z, b):
        
        specmodel = np.ones_like(self.obj_wl_over)
        
        voigt=self.voigt_model(self.obj_wl_over, N, b, z, self.linefill['wrest'].value, self.linefill['f'], self.linefill['gamma'].value) 
        specmodel *= voigt
            
        #return the re-normalized model normalized by continuum
        if specresolution > self.velstep:
             specmodel = self.convolve_model(specmodel, specresolution)
        
        #Now rebin to match original wl array and return
        specmodel *= continuum
        
        return np.mean(specmodel.reshape(len(self.obj),self.oversample), axis=1)
    
    def reconstruct_spec(self, p, targonly=False):
        
        #targonly means reconstruct the full profile of the lines without fillers
        if self.freespecres:
           specresolution = p[0]
        else:
           specresolution = self.specres   
        
        if self.freecont:
           if self.freespecres:
              continuum = p[1]
           else:
              continuum = p[0]   
        else:
           continuum = self.contval
        
        specmodel = np.ones_like(self.obj_wl_over)
        thisncomp = int(p[self.startind])
        
        for comp in range(thisncomp):
            _N, _z, _b = p[1+3*comp+self.startind:1+3*comp+3+self.startind] #First one is for Ncomp in the fit
            
            for line in range(self.numlines):
               voigt=self.voigt_model(self.obj_wl_over, _N, _b, _z, self.linepars[line]['wrest'].value, self.linepars[line]['f'], self.linepars[line]['gamma'].value) 
               specmodel *= voigt
        
        if not targonly:
           for fill in range(self.nfill):
              _N, _z, _b = p[3*fill+self.endind:3*fill+3+self.endind]
             
              voigt=self.voigt_model(self.obj_wl_over, _N, _b, _z, self.linefill['wrest'].value, self.linefill['f'], self.linefill['gamma'].value) 
              specmodel *= voigt
            
        #return the re-normalized model normalized by continuum
        if specresolution > self.velstep:
             specmodel = self.convolve_model(specmodel, specresolution)
        
        #Now rebin to match original wl array and return
        specmodel *= continuum
        
        return np.mean(specmodel.reshape(len(self.obj),self.oversample), axis=1)
            
    
    def convolve_model(self, spec, fwhm):
    
        sigma = self.oversample* (fwhm / 2.354820) / (self.velstep)
        # gaussian drops to 1/100 of maximum value at x =
        # sqrt(2*ln(100))*sigma, so number of pixels to include from
        # centre of gaussian is:
        n = np.ceil(3.0348 * sigma)
        x_size = int(2*n) + 1 
        #return scipy_convolve(spec, Gaussian1DKernel(sigma, x_size=x_size),
        #                mode='same', method='direct')

        return convolve(spec, Gaussian1DKernel(sigma, x_size=x_size),
                        boundary='wrap', normalize_kernel=True)
    
    
    
    
    def calc_ew(self, p, lineid=0):
        
        #Calculate rest frame equivalent width of the 
        #absorption profile
        Wtot = 0

        if self.freecont:
           if self.freespecres:
             cont = p[1]
           else:
             cont = p[0]  
        else:
           cont = self.contval
        
        for comp in range(self.ncompmax):
            _N, _z, _b = p[3*comp+self.startind:3*comp+3+self.startind]
            absorption=(np.zeros_like(self.obj)+cont)*self.voigt_model(self.obj_wl, _N, _b, _z, self.linepars[lineid]['wrest'].value, self.linepars[lineid]['f'], self.linepars[lineid]['gamma'].value) 
            
            dlambda = np.diff(self.obj_wl)
            dlambda = np.insert(dlambda,0,dlambda[0])
            
            Wtemp = np.sum( (1-(absorption/cont)) * dlambda )
            Wtot += Wtemp/(1+_z)
        
        return Wtot
    
    def calc_N(self, p):
        
        #Calculate column density of the 
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
      
      print('Sorting components in redshift')
      postsorted = np.copy(postsamples) 
      ncols= len(postsorted[0])
      startind = (ncols-1) %3
      
      for ii in range(len(postsamples[:,0])):
         
         thisncomp = int(postsamples[ii,startind])
         thisendind = startind+1+3*thisncomp
         
         #Set all values above thisncomp to 99 both in original array and in sorted one
         postsamples[ii,thisendind:] = 99
         postsorted[ii,thisendind:] = 99
         
         zsort = np.argsort(postsamples[ii,startind+2:startind+1+3*thisncomp:3])
         for jj in range(len(zsort)):
                postsorted[ii,3*jj+startind+1:3*jj+3+startind+1] = postsamples[ii,3*zsort[jj]+[0,1,2]+startind+1]
         
         postsorted[postsorted==99] = np.nan
            
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
          for i in range(int(nwaves/2)):
            wavefit.append((float(tmpwavefit[2*i]),float(tmpwavefit[2*i+1])))  
        
    if not input_params.has_option('input', 'linelist'):
        raise configparser.NoOptionError("input", "linelist")
    else:
        linelist = input_params.get('input', 'linelist').split(',')
        
    if not input_params.has_option('input', 'coldef'):
        coldef = ['Wave', 'Flux', 'Err']
    else:
        coldef = input_params.get('input', 'coldef').split(',') 
    
    if input_params.has_option('input', 'specres'):
       specres =  np.array(input_params.get('input', 'specres').split(','), dtype=float)
    else:
       specres = np.array(([7.0]), dtype=float)

    if input_params.has_option('input', 'asymmlike'):
        asymmlike = booldir[input_params.get('input', 'asymmlike')]
    else:
        asymmlike = False       
        
    if input_params.has_option('input', 'solver'):
        solver = input_params.get('input', 'solver')
    else:
        solver = 'polychord'    
        
    if input_params.has_option('input', 'oversample'):
        oversample = int(input_params.get('input', 'oversample'))
    else:
        oversample = 1   
    
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
       contval =  np.array(input_params.get('components', 'contval').split(','), dtype=float)
    else:
       contval = np.array([1])
       
    if input_params.has_option('components', 'Nrange'):
       Nrange = np.array(input_params.get('components', 'Nrange').split(','), dtype=float)
    else:
       Nrange = np.array((11.5,16))

    if input_params.has_option('components', 'brange'):
       brange = np.array(input_params.get('components', 'brange').split(','), dtype=float)
    else:
       brange = np.array((1,30))

    if input_params.has_option('components', 'zrange'):
       zrange = np.array(input_params.get('components', 'zrange').split(','), dtype=float)
    else:
       zrange = None

    if input_params.has_option('components', 'Nrangefill'):
       Nrangefill = np.array(input_params.get('components', 'Nrangefill').split(','), dtype=float)
    else:
       Nrangefill = np.array((11.5,16))

    if input_params.has_option('components', 'brangefill'):
       brangefill = np.array(input_params.get('components', 'brangefill').split(','), dtype=float)
    else:
       brangefill = np.array((1,30))
       
    if input_params.has_option('components', 'wrangefill'):
       wrangefill = np.array(input_params.get('components', 'wrangefill').split(','), dtype=float)
    else:
       wrangefill = None
       
    if input_params.has_option('plots', 'nmaxcols'):
       nmaxcols = int(input_params.get('plots', 'nmaxcols')[0])
    else:
       nmaxcols = 5

    if input_params.has_option('plots', 'yrange'):
       yrange = np.array(input_params.get('plots', 'yrange').split(','), dtype=float)
    else:
       yrange = np.array((-0.1,1.2))
    
    #Parameters driving the run
    if input_params.has_option('run', 'dofit'):
       dofit = booldir[input_params.get('run', 'dofit')]
    else:
       dofit = True

    if input_params.has_option('run', 'doplot'):
       doplot = booldir[input_params.get('run', 'doplot')]
    else:
       doplot = True
       
       

    run_params = {'specfile'  : datadir+input_params.get('input', 'specfile'),
                  'wavefit'   : wavefit,
                  'linelist'  : linelist,
                  'coldef'    : coldef,
                  'asymmlike' : asymmlike,
                  'solver'    : solver,
                  'oversample': oversample,
                  'specres'   : specres,
                  'chaindir'  : chaindir,
                  'plotdir'   : plotdir,
                  'chainfmt'  : chainfmt,
                  'ncomp'     : ncomp,
                  'nfill'     : nfill,
                  'Nrange'    : Nrange,
                  'brange'    : brange,
                  'zrange'    : zrange,
                  'Nrangefill': Nrangefill,
                  'brangefill': brangefill,
                  'wrangefill': wrangefill,
                  'contval'   : contval,
                  'nmaxcols'  : nmaxcols,
                  'yrange'    : yrange,
                  'dofit'     : dofit,
                  'doplot'    : doplot}
                  
    if input_params.has_section('mnsettings'):
    
       settingsdict = (dict((opt, booldir[input_params.get('mnsettings',opt)]) 
                 if input_params.get('mnsettings',opt) in ['True','False'] 
                 else (opt, input_params.get('mnsettings', opt)) 
                 for opt in input_params.options('mnsettings')))
    
       run_params['mnsettings'] = settingsdict


    if input_params.has_section('pcsettings'):
    
       settingsdict = (dict((opt, booldir[input_params.get('pcsettings',opt)]) 
                 if input_params.get('pcsettings',opt) in ['True','False'] 
                 else (opt, input_params.get('pcsettings', opt)) 
                 for opt in input_params.options('pcsettings')))
    
       run_params['pcsettings'] = settingsdict
                  
    

    return run_params

#b /cosma/home/durham/dc-foss1/.local/lib/python3.7/site-packages/mc_alf-0.99-py3.7.egg/mcalf/routines/hires_fitter.py:558#
#b /home/matteo/.local/lib/python3.8/site-packages/mc_alf-0.99-py3.8.egg/mcalf/routines/hires_fitter.py:82
