# Configuration File for SpecMod


SPECTRAL = dict(

# BINNING PARAMS
## min freq, max freq, # of bins
BIN_PARS = (0.0001, 200, 151),

# SNR PARAMS
## minimum SNR required for pass
SNR_TOLERENCE = 2,
## minimum number of points above SNR to pass
MIN_POINTS = 5,
## bands to evaluate SNR (like Shearer)
S_BANDS = [(9, 10),],

ROTATE = True,
# SNR METHOD
BW_METHOD = 2,
## rotation params if BW_METHOD=2
ROT_PARS = (0, -1, 0.001),

# VIS OPTS
PLOT_COLUMNS = 3

)

MODELS = dict(

MODEL = "BRUNE",
MOTION = "velocity"

)

FITTING = dict(
# HOW MANY COLS TO PLOT
PLOT_COLUMNS = 3
)
