import os
import glob
import obspy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.dates import num2date
import numpy as np
import pandas as pd


# def get_filt_params(x):
#     return x.split("options=")[-1].split("::")[0].strip("{").strip("}")
#
# get_filt_params(sig[0].stats.processing[2])



def plot_traces(st, plot_theoreticals=False, plot_windows=False, conv=1e-9,
                    bft=1, aftt=60, sig=None, noise=None, save=None):

    sharey=False
    stc = stream_distance_sort(st)
    if conv is None:
        conv=1
        stc.normalize()
        sharey=True

    if sig is not None and noise is not None:
        sig = stream_distance_sort(sig)
        noise = stream_distance_sort(noise)

    fig, ax = plt.subplots(len(stc), 1, sharex=True, sharey=sharey, figsize=(8, 14))
    ax = ax.flatten()

    for i, tr in enumerate(stc):

        if plot_windows:
            # get window start and end times
            sts = sig[i].stats['wstart'], sig[i].stats['wend']
            nts = noise[i].stats['wstart'], noise[i].stats['wend']

            tr.trim(nts[0]-bft, sts[1]+aftt)

            for ts, tn in zip(sts, nts):
                ts, tn = num2date(ts.matplotlib_date), num2date(tn.matplotlib_date)
                ax[i].vlines(ts,tr.data.min()*conv,tr.data.max()*conv,color='k')
                ax[i].vlines(tn,tr.data.min()*conv,tr.data.max()*conv,color='blue')

        if plot_theoreticals:

            if not plot_windows:
                # we should trim it down to some time before and after the p arrival (assuming we trust it)
                tr.trim(tr.stats['p_time']-bft, tr.stats['p_time']+aftt)

            p = num2date(tr.stats['p_time'].matplotlib_date)
            s = num2date(tr.stats['s_time'].matplotlib_date)
            ax[i].vlines(p, tr.data.min()*conv/1.5, tr.data.max()*conv/1.5,
                linestyles='dashed', color='blue', label='Pg')
            ax[i].vlines(s, tr.data.min()*conv/1.5, tr.data.max()*conv/1.5,
                linestyles='dashed', color='red', label='Sg')

        time = num2date(np.array([
            (tr.stats.starttime+(tr.stats.delta*(
                i+1))).matplotlib_date for i in range(len(tr.data))]))

        ax[i].plot(time, tr.data*conv, color='grey', label=tr.id, zorder=1)
        try:
            ax[i].set_title("Repi: {:.2f} km, Rhyp: {:.2f} km".format(
                tr.stats['repi'], tr.stats['rhyp']))
        except KeyError:
            pass
        ax[i].legend()
    ax[-1].set_xlabel('Time (UTC)')
    fig.suptitle(str(st[0].stats.otime))
    fig.tight_layout()
    if save is not None:
        fig.savefig(os.path.join("Figures",save))

def stream_distance_sort(st, dist_met='repi'):
    """
    Sorted makes a copy so you must save it back to rhe stream
    then force the user to save a new copy. NOT INPLACE!!!
    """
    try:
        st = obspy.Stream(sorted(st, key=lambda x: x.stats[dist_met]))
    except KeyError:
        print('WARNING: No distance info, stream not sorted by distance.')

    return st


def cps(st):
    return st.copy()

class DataSet():

    pdir=" "
    pref=('HH','BH','EN')
    stat_pref=('BRWY','GAWY','FMC','SVWY')
    allpaths=list()
    stations=list()
    available=list()
    filtered=list()
    pref_sta=True

    def __init__(self, path, pdir="Data", pref_sta=True):
        self.pdir=os.path.join(pdir, path)
        self.pref_sta = pref_sta
        self.__startup()


    def get_obs_paths(self):

        apaths = [os.path.join(self.pdir, ".".join(t)) for t in zip(*self.available)]

        if self.pref_sta:
            return [p for p in apaths if p.split(".")[-2] in self.stat_pref]
        else:
            return apaths

    def __get_paths(self):
        tmp = glob.glob(os.path.join(self.pdir, "*.sac"))
        self.allpaths = [d.split("/")[-1] for d in tmp]
        print("found {} files in {}.".format(len(self.allpaths), self.pdir))

    def __get_stations(self):
        self.stations = sorted(list(set([x.split(".")[-3] for x in self.allpaths])))

    def __set_ranks(self):
        self.ranks = rank_chans(self.pref)

    def __get_available(self):
        self.available = get_avail(self.ranks, self.allpaths)

    def __startup(self):
        self.__get_paths()
        self.__get_stations()
        self.__set_ranks()
        self.__get_available()



def getchan(path):
    return path.split(".")[-2]

def rank_chans(prefs):
    num = [x+1 for x in range(len(prefs))]
    ranks = {}
    for p, n in zip(prefs, num):
        ranks.update({p:n})
    return ranks

def compare_ranks(ranks, current, new):
    return ranks[current] < ranks[new]

def get_avail(ranks, allpaths):
    fs, chan = [], []
    for f in allpaths:
        if getchan(f)[:2] in ranks.keys():
            tmp = ".".join(f.split(".")[:-2])
            if tmp in fs:
                ind = fs.index(tmp)
                new = getchan(f)[:2]
                if compare_ranks(ranks, chan[ind], new):
                    chan[ind]=new

            else:
                fs.append(tmp)
                chan.append(getchan(f)[:2])
    return fs, chan



def read_cat(path):
    return pd.read_csv(path, delim_whitespace=True)



def cat2kstyle(row):
    d = row['Date'].replace("/", ".")
    t = row['Time'].replace(":", ".")[:-3]
    return ".".join([d, t])


def keith2utc(row):
    return obspy.UTCDateTime((*list(map(int, cat2kstyle(row).split(".")))))


def path_to_utc(p):
    return obspy.UTCDateTime(*list(map(int, p.split(".")[1:])))
