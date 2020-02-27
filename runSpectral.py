import os
import specmod.PreProcess as pre
import specmod.utils as ut
import specmod.Spectral as sp
import specmod.Fitting as fit
import specmod.Models as md


def trona_preprocess(st, p):
    stc=st.copy()
    pre.set_origin(stc, ut.path_to_utc(p))
    stc.detrend()
    stc.taper(0.05)
    stc.filter("highpass", freq=0.2, zerophase=False)
    return stc


if __name__ == "__main__":


    cat = ut.read_cat('Data/templates.cat')
    paths = sorted([d for d in os.listdir("Data") if d[0] in ["D", "T"]])

    for p in paths:
        if p[0] == "T":
            # use actual origin for templates
            row = cat[(
                cat.Date == "/".join(p.split(".")[1:4]))&(
                cat.Time.str[0:-3] == ":".join(p.split(".")[4:]))]
            olat, olon, odep = row.Lat.values[0], row.Lon.values[0], row.Dep.values[0]

        if p[0] == "D":
            # use the centroid origin for detections
            olat, olon, odep = cat.Lat.values.mean(), cat.Lon.values.mean(), cat.Dep.values.mean()

        sta_shift={'GAWY':0.6, 'FMC':-0.5}

        d = ut.DataSet(os.path.join(p, 'rr'))

        st = pre.rstfl(d.get_obs_paths(), wild='Z')

        for tr in st:
            print(tr.stats.sac)

        st2 = trona_preprocess(st, p)

        pre.set_stream_distance(st2, olat, olon, odep, dtype="sac")

        pre.basic_set_theoreticals(st2, ut.path_to_utc(p), p=4.2, s=2.6)

        sig = pre.get_signal(st2, pre.cut_p, bf=0.5, raf=0.8, sta_shift=sta_shift)

        noise = pre.get_noise_p(st2, sig, bshift=0.7, bf=3)
        # visualise spectra as sanity check
        # ut.plot_traces(st2, plot_theoreticals=True, bft=5, aftt=30)

        # ut.plot_traces(st2, plot_theoreticals=True, bft=5,
        #     aftt=10, plot_windows=True, sig=sig, noise=noise, conv=1)
        #     #save=".".join(("Trona", str(ut.keith2utc(row)), "pdf")))
        #
        # ut.plot_traces(sig) #, save=".".join(("Trona", "psig", str(ut.keith2utc(row)),
            #"pdf")))

        # snps = sp.calc_spectra(sig, noise, number_of_tapers=5, quadratic=True)
        snps = sp.Spectra.from_streams(sig, noise, number_of_tapers=5, quadratic=True)

        snps.psd_to_amp()

        snps.write_spectra(os.path.join("Output/Spectra",str(ut.path_to_utc(p))), snps)


    itr = "2"
    for spec in [os.path.join("Output/Spectra", s) for s in os.listdir("Output/Spectra") if s.endswith(".spec")]:

        snps = sp.Spectra.read_spectra(spec, "pickle",
            skip_warning=True)

        guess = fit.FitSpectra.create_simple_guess(snps)

        fits = fit.FitSpectra(snps, md.simple_model, guess)
        fits.set_bounds('ts', min=0.001)

        # fits.get_fit("UU.GAWY.01.HHZ").result.params
        if itr == "2":
            import numpy as np
            try:
                db = fits.read_flatfile(os.path.join(f"Output/itr_1/Flatfiles", fits.spectra.event+".csv"))
                fc = 10**db.fc.apply(np.log10).describe()["75%"]
                print(f"fc fixed at {fc} Hz for {fits.spectra.event}")
                fits.set_const('fc', fc)
            except Exception as msg:
                print(f"skipping {fits.spectra.event}")
                continue
        # fits.reset()
        fits.fit_spectra('log', method='powell')
        snps.quick_vis(save=os.path.join(f"itr_{itr}", f"Trona_psig_spectra{fits.spectra.event}.svg"))

        fits.write_flatfile(os.path.join(f"Output/itr_{itr}/Flatfiles", fits.spectra.event+".csv"), fits)



    import pandas as pd

    df1 = pd.DataFrame([])
    for p in [os.path.join(f"Output/itr_{1}/Flatfiles", d) for d in os.listdir(f"Output/itr_{1}/Flatfiles") if d.endswith(".csv")]:
        df1 = pd.concat((df1, pd.read_csv(p)), ignore_index=True, sort=False)

    df2 = pd.DataFrame([])
    for p in [os.path.join(f"Output/itr_{2}/Flatfiles", d) for d in os.listdir(f"Output/itr_{2}/Flatfiles") if d.endswith(".csv")]:
        df2 = pd.concat((df2, pd.read_csv(p)), ignore_index=True, sort=False)

    df2.plot('repi', 'llpsp', kind='scatter')

    

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(3, 1, figsize=(5, int(3*3.5)), sharex=True)
    for df, c, l in zip((df1, df2), ("k", "r"), ("fc free", "fc fixed")):
        ax[0].plot(df.repi[df.llpsp<=df.llpsp.describe()["75%"]], df.llpsp[df.llpsp<=df.llpsp.describe()["75%"]], f'{c}o', mfc='none', label=l)
        ax[1].plot(df.repi[df.fc<=df.fc.describe()["75%"]], df.fc[df.fc<=df.fc.describe()["75%"]], f'{c}o', mfc='none')
        ax[2].plot(df.repi[df.ts<=df.ts.describe()["75%"]], df.ts[df.ts<=df.ts.describe()["75%"]], f'{c}o', mfc='none')
    ax[0].legend()
    ax[0].set_ylabel("Log Long Period Spectral Amplitude [m/s]/s")
    ax[1].set_ylabel("Corner Frequency [Hz]")
    ax[2].set_ylabel("t* (whole path attenuation) [Hz]")
    ax[-1].set_xlabel("Epicentral Distance [km]")

    fig.savefig("Figures/Spectral_Modelling_Summary.svg")
