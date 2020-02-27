import numpy as np

BRUNE_MODEL = (1, 2)
BOATRIGHT_MODEL = (2, 2)

MODEL = BRUNE_MODEL
motion='velocity'


def scale_to_motion(motion, f):
    if motion.lower() == 'displacement':
        return 0

    elif motion.lower() == 'velocity':
        return np.log10(2*np.pi*f)

    elif motion.lower() == 'acceleration':
        return np.log10(np.power(2*np.pi*f,2))
    else:
        return None


def source(f, llpsp, fc):
    gam, n = MODEL
    loga = llpsp - (1/gam)*np.log10((1+(f/fc)**(gam*n)))
    return loga

def atten(f, ts):
    return -(np.pi*f*ts / np.log(10))

def simple_model(f, llpsp, fc, ts):
    global motion
    """
    Simple attenuated source model to minimise.
    """
    return source(f, llpsp, fc) + atten(f, ts) + scale_to_motion(motion, f)




# import matplotlib.pyplot as plt
# f = np.arange(0.01, 100, 0.01)
# a = source(1, 1, f) + np.log10(scale_to_motion('displacement', f))
# plt.semilogx(f, a)
