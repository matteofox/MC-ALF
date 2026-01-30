# MC-ALF: Monte-Carlo Absorption Line Fitter

MC-ALF is a Python package designed for fitting absorption lines in spectra, particularly Voigt profiles, using Bayesian, Monte-Carlo techniques. It leverages nested sampling algorithms to explore the parameter space and find the best-fit parameters for column densities, redshifts, and Doppler parameters.

From V2.0 it runs with the JAX solver, and has been tested to run on GPUs using CUDA13.

This project has been supported by the European Union – Next Generation EU, Mission 4, Component 1 CUP H53D23011030001.

## Installation

### Prerequisites

MC-ALF requires Python 3 and the following dependencies:

*   numpy
*   scipy
*   astropy
*   matplotlib
*   mpi4py
*   linetools
*   (Optional) jax, jaxns, tensorflow_probability (for JAX-based solving)
*   (Optional) pypolychord (for PolyChord solver)
*   (Optional) pymultinest (for MultiNest solver)
*   (Optional) dynesty (for Dynesty solver)

### Installing from Source

You can install the package directly from the source code:

```bash
git clone https://github.com/matteofox/mc-alf.git
cd mc-alf
pip install .
```

This will install the package and create the `mc-alf` command in your environment.

## Usage

The main interface to MC-ALF is the command-line script `mc-alf`. It takes a configuration file as an argument.

```bash
mc-alf config_file.ini
```

### Options

*   `--debug`: Enable debug mode for increased verbosity.

## Configuration File Guide

The configuration file is an INI-style file that controls the fitting process. Below are the key sections and parameters.

### [input]

*   `specfile`: Full path to the spectrum file (ASCII format expected with columns for Wave, Flux, Error).
*   `wavefit`: Comma-separated list of wavelength ranges to fit. Must be in pairs (min, max). Example: `1210,1220, 1545,1555`.
*   `linelist`: Comma-separated list of lines to fit (standard `linetools` names). Example: `HI 1215, CIV 1548`.
*   `coldef`: (Optional) Column names in the input file. Default: `Wave, Flux, Err`.

### [parameters]

*   `ncomp`: Tuple defining the minimum and maximum number of components to fit. Example: `(1, 3)`.
*   `dofit`: Boolean. Set to `True` to perform the fit.
*   `doplot`: Boolean. Set to `True` to generate plots after fitting.
*   `solver`: The nested sampling solver to use. Options: `polychord`, `multinest`, `dynesty`, `jaxns`.

### [priors]

*   `nfill`: Number of "filler" components (nuisance lines) to include.
*   `contval`: Initial continuum value (or fixed value if not floating).
*   `specres`: Spectral resolution (FWHM in km/s). Can be a list `[min, max]` to float, or a single value to fix.
*   `Nrange`: Log column density range `[min, max]` for target lines.
*   `brange`: Doppler parameter range `[min, max]` (km/s) for target lines.
*   `zrange`: Redshift range. If omitted, defaults to the range covered by the spectrum for the first line.
*   `Nrangefill`: Log column density range for filler lines.
*   `brangefill`: Doppler parameter range for filler lines.
*   `wrangefill`: Wavelength range (in Angstroms) for filler lines to exist.

### [output]

*   `chaindir`: Directory to save the output chains.
*   `plotdir`: Directory to save the output plots.
*   `chainfmt`: Format string for output filenames.
*   `nmaxcols`: Maximum number of columns in the output plot.

### Solver Settings
solver-specific settings can be provided in sections like `[pc_settings]` (PolyChord), `[mnsettings]` (MultiNest), or `[jaxns_settings]`.

## Output

### Data Products
The code generates several output files in the `chaindir`:
*   `*_equal_weights.txt`: Posterior samples with equal weights. Columns are usually: `weight`, `-2*logL`, `param1`, `param2`, ...
*   `*.stats`: Contains Bayesian evidence (logZ) and other statistics.

### Plots
If `doplot = True`, PDF plots are generated in `plotdir` showing the spectrum, the best-fit model (red), and individual components (blue for target, red/dotted for fillers).

## Example Configuration

```ini
[input]
specfile = testdata/civ_mock_spec_multicomp.txt
wavefit = 6180,6220
linelist = CIV 1548, CIV 1550
coldef = Wave, Flux, Err
solver = polychord
specres = 8.0,9.0
asymmlike = False

[pathing]
datadir = ./
outdir = testdata/output/
chainfmt = pc_fits_{0}
chaindir = pc_fits/
plotdir  = pc_plots/

# ncomp can be a range, nfill is currently a fixed value
# If not specified we assume one component and zero fillers
[components]
ncomp = 8,11
contval  = 1
Nrange = 12.0,14.5
brange = 10.0, 40.0
zrange = 2.99, 3.01
Nrangefill = 11.5,16
brangefill = 1,30

[run]
dofit = False
doplot = True
device = gpu 
showprogress=True

[jaxns_settings]
max_samples = 20000
num_live_points = 200

[pc_settings]
nlive = 150
num_repeats = 25
precision_criterion = 0.01
feedback = 1
do_clustering = False

[plots]
nmaxcols = 3
```
