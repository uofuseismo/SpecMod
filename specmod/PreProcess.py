import os
import glob
import obspy
import matplotlib.pyplot as plt
import numpy as np


STREAM_DISTANCE_METHODS = ["mseed", "sac", "none"]

def set_origin(st, ot):
    for tr in st:
        tr.stats['otime'] = ot


def set_stream_distance(st, olat, olon, odep, stlats=None, stlons=None, stelvs=None, inventory=None, dtype="sac"):

    """
    This assumes the sac files already have the station lat and long (degrees)
    and elevation in the sac header.
    """

    global STREAM_DISTANCE_METHODS

    if dtype.lower() in STREAM_DISTANCE_METHODS:

        for i, tr in enumerate(st):
            tr.stats['dep'] = odep
            tr.stats['olon'] = olon
            tr.stats['olat'] = olat
            if dtype.lower() == "sac":
                tr.stats['repi'] = obspy.geodetics.gps2dist_azimuth(tr.stats.sac.stla, tr.stats.sac.stlo, olat, olon)[0]/1000
                tr.stats['rhyp'] = np.sqrt((odep+(tr.stats.sac.stel/1000))**2+tr.stats['repi']**2)
                tr.stats['slon'] = tr.stats.sac.stlo
                tr.stats['slat'] = tr.stats.sac.stla
                tr.stats['selv'] = tr.stats.sac.stel
            elif dtype.lower() == "mseed":
                stlat, stlon, stelv = get_station_coords_from_inventory(tr, inv)
                tr.stats['slon'] = stlon
                tr.stats['slat'] = stlat
                tr.stats['selv'] = stelv
                tr.stats['repi'] = obspy.geodetics.gps2dist_azimuth(stlat, stlon, olat, olon)[0]/1000
                tr.stats['rhyp'] = np.sqrt((odep+(stelv/1000))**2+tr.stats['repi']**2)

            elif dtype.lower() == "none":
                stlat, stlon, stelv = stlats[i], stlons[i], stelvs[i]
                tr.stats['slon'] = stlon
                tr.stats['slat'] = stlat
                tr.stats['selv'] = stelv
                tr.stats['repi'] = obspy.geodetics.gps2dist_azimuth(stlat, stlon, olat, olon)[0]/1000
                tr.stats['rhyp'] = np.sqrt((odep+(stelv/1000))**2+tr.stats['repi']**2)

            else:
                print("invalid method choice")


def get_station_coords_from_inventory(tr, inv):
    pass
    return slat, slon, selev

def basic_set_theoreticals(st, otime, p=5.9, s=2.9, dmetric='repi'):
    """
    basic_set_theoreticals uses average propagation velocities [km/s]
    to set the arrival times for P and S waves. This assumes epicentral and/or
    and hypocentral distances have already been calculated and are set in the
    trace stats dictionary as tr.stats['repi'] or tr.stats['rhyp'] in units of
    kilometres.
    """
    for tr in st:
        rel_p = tr.stats[dmetric]/p
        rel_s = tr.stats[dmetric]/s
        tr.stats['p_time'] = otime+rel_p
        tr.stats['s_time'] = otime+rel_s
        tr.stats['otime'] = otime


def rstfl(fnames, wild="*", ext="sac"):
    """
    rstfl reads create an obspy stream by reading each trace from an arbitrary
    list of paths.
    """

    st = obspy.Stream([])
    for f in fnames:
        st += obspy.read("{}{}.{}".format(f, wild, ext.lower()), format=ext.upper())

    return st



def link_window_to_trace(tr, start, end):
    tr.stats['wstart'] = start
    tr.stats['wend'] = end


def get_sta_shift(sta, sta_shift):
    """
    sta_shift must be a dictionary containing the station name to be shifted
    and the time shift in seconds e.g. {'STA':0.5}.
    """
    if sta in sta_shift.keys():
        return sta_shift[sta]
    else:
        return 0

def cut_p(st, bf=2, raf=0.8, sta_shift=dict()):
    """
    Function to cut a p wave window from an Obspy trace obeject
    """
    stas=0

    for tr in st:

        stas = get_sta_shift(tr.stats.station, sta_shift)
        relps = tr.stats['s_time'] - tr.stats['p_time']
        p_start = tr.stats['p_time']-bf+stas
        p_end = tr.stats['p_time']+relps*raf

        link_window_to_trace(tr, p_start, p_end)
        tr.trim(p_start, p_end)

def get_signal(st, func, **kwargs):
    stc = st.copy()
    func(stc, **kwargs)
    return stc

def get_noise_p(st, sig, bf=1, bshift=0.2):
    stc=st.copy()
    for tr, trs in zip(stc, sig):
        end = trs.stats['wstart']-bshift
        start = end - bf
        link_window_to_trace(tr, start, end)
        tr.trim(start, end)
    return stc
