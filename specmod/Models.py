# MODELS contains a set of functions for minimisation to seismic spectra.
# It can be modified as appropriate.
import numpy as np
from . import config as cfg

MODS = ["BRUNE", "BOATWRIGHT"]

# UTIL FUNCS
def which_model(mod):
    if mod in MODS:
        if mod == "BRUNE":
            return BRUNE_MODEL
        if mod == "BOATWRIGHT":
            return BOATWRIGHT_MODEL
    else:
        raise ValueError(f"Model {mod} not available. Choose from {MODS}.")


def scale_to_motion(motion, f):
    if motion.lower() == 'displacement':
        return 0

    elif motion.lower() == 'velocity':
        return np.log10(2*np.pi*f)

    elif motion.lower() == 'acceleration':
        return np.log10(np.power(2*np.pi*f,2))
    else:
        return None

# DEFAULT PARAMS FOR SOURCE MODE:
BRUNE_MODEL = (1, 2) # omega squared
BOATWRIGHT_MODEL = (2, 2) # omega cubed
#
MODEL = which_model(cfg.MODELS["MODEL"])
MOTION = cfg.MODELS["MOTION"]



# MINIMISATION FUNCTIONS
## Source model
def source(f, llpsp, fc):
    gam, n = MODEL
    loga = llpsp - (1/gam)*np.log10((1+(f/fc)**(gam*n)))
    return loga

# t-star attenuation model
def atten(f, ts):
    return -(np.pi*f*ts / np.log(10))

# combine models
def simple_model(f, llpsp, fc, ts):
    global MOTION
    """
    Simple attenuated source model to minimise.
    """
    return source(f, llpsp, fc) + atten(f, ts) + scale_to_motion(MOTION, f)
