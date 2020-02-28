import os
import numpy as np
import lmfit as lm
import pandas as pd
import specmod.Spectral as sp
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
from matplotlib.ticker import StrMethodFormatter, NullFormatter

# import specmod.Models as md



# abrune = Model(simple_model)
# params = abrune.make_params(llpsp=-10, fc=3, ts=0.01)
# params['fc'].vary = True
# result = abrune.fit(np.log10(amp), params, f=f, method='tnc', weights=(1/f)/(1/f).max())
#
# plt.semilogx(f, np.log10(amp), '.', color='grey', mfc='none')
# plt.semilogx(f, result.init_fit, 'k', label='inital')
# plt.semilogx(f, result.best_fit, 'r--', label='final')
#
# result
# help(params)
# dir(params['llpsp'])


class FitSpectrum(object):

    """
    Takes an Spectral.Signal and fits an arbitrary model to the signal spectrum
    using the lmfit package.
    """

    sig = sp.Signal()
    mod = None
    params = None
    result = None
    mod_freq = np.array([])
    mod_amp = np.array([])
    pass_fitting=True
    meta = {}

    def __init__(self, signal, model, **params):
        self.set_signal(signal)
        self.set_model(model, **params)

    def fit_mod(self, **kwargs):
        self.result = self.mod.fit(
            self.mod_amp, self.params, f=self.mod_freq, **kwargs)
        self.__set_results_to_meta()
        self.__determine_pass_or_fail()

    def set_signal(self, signal):
        if self.__check_input(signal):
            self.sig = signal
            self.__set_meta(signal.meta)
        # if setting a new signal - assess and adjust the freq bounds
        self.__set_mod_amp_freq()

    def set_model(self, model, **params):
        self.mod = lm.Model(model)
        # whenever a model is set the inital params must be set also
        self.__init_params(**params)

    def set_const(self, pname, value):
        self.params[pname].value = value
        self.params[pname].vary = False

    def set_bounds(self, pname, min=None, max=None):
        if min is not None:
            self.params[pname].min = min
        if max is not None:
            self.params[pname].max = max

    def __set_meta(self, meta):
        self.meta = deepcopy(meta)


    def __init_params(self, **params):
        self.params = self.mod.make_params(**params)
        # self.set_bounds('fc', min=0)

    def reset(self):
        for par in self.params.values():
            par.vary = True
            par.min = -np.inf
            par.max = np.inf

    def __check_input(self, signal):
        if type(signal) is not type(sp.Signal()):
            raise ValueError(
                "Must be a signal object not {}".format(type(signal)))
        else:
            return True

    def __set_mod_amp_freq(self):
        """
        Only fit between signal limits if they are specified.
        """
        if self.sig.ubfreqs.size > 0:
            inds = np.where((self.sig.freq>=self.sig.ubfreqs[0]) & (
                 self.sig.freq<=self.sig.ubfreqs[1]))
            self.mod_freq = self.sig.freq[inds]
            self.mod_amp = self.sig.amp[inds]
        else:
            self.mod_freq = self.sig.freq
            self.mod_amp = self.sig.amp
        self.mod_amp = np.log10(self.mod_amp)


    def __param_string(self):
        pars = [[k.name, k.value, 2*k.stderr] for k in self.result.params.values()]
        return ", ".join(['{}: {:.3f}+/-{:.3f}' for _ in pars]).format(
            *[val for sublist in pars for val in sublist])

    def quick_vis(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(1,1)

        ax.loglog(self.mod_freq, 10**self.mod_amp, color='grey', label=self.sig.id)
        ax.loglog(self.mod_freq, 10**self.result.best_fit, 'k--', label='model')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(self.__param_string())
        ax.set_xlabel('freq [Hz]')
        ax.set_ylabel('spectral amp')
        ax.legend()

        if ax is not None:
            return ax

    def __get_pars(self):
        p={}
        for k in self.result.params.values():
            p.update({k.name: k.value})
            p.update({k.name+"-std":k.stderr})
        return p

    def __get_fit_stats(self):
        res = self.result
        s = {}
        s.update({'aic':res.aic})
        s.update({'bic':res.bic})
        s.update({'chisqr':res.chisqr})
        s.update({'redchi':res.redchi})
        return s

    def __get_test_results(self):
        t = {}
        t.update({'pass_fitting': self.pass_fitting})
        return t

    def __set_results_to_meta(self):
        self.meta.update(self.__get_pars())
        self.meta.update(self.__get_fit_stats())
        self.meta.update(self.__get_test_results())
    def __determine_pass_or_fail(self):
        for par, vals in self.result.params.items():
            try:
                if (vals.value-vals.stderr <= vals.min) or (vals.value+vals.stderr >= vals.max):
                    # print(par, vals)
                    self.pass_fitting = False
            except TypeError:
                # print("std err is none")
                # print(par, vals)
                self.pass_fitting = False



class FitSpectra(object):
    spectra = sp.Spectra()
    models = {}
    guess = {}
    table = pd.DataFrame([])

    def __init__(self, spectra, model, guess=None):
        self.set_spectra(spectra)
        if guess is not None:
            self.init_fitting(model, guess)


    def __len__(self):
        return len(self.models)

    def set_spectra(self, spectra):
        if self.__check_spectra(spectra):
            self.spectra = spectra

    def get_spectra(self):
        return self.spectra

    def get_fit(self, id):
        if id.upper() in self.models.keys():
            return self.models[id.upper()]
        else:
            print("WARNING: {} not in group of available fits.".format(id.upper()))

    def fit_spectra(self, weight_method='none', **kwargs):
        wm = self.__check_wm(weight_method)
        for name, mod in self.models.items():
            try:
                if wm == 'log':
                    mod.fit_mod(weights=1/mod.mod_freq, **kwargs)
                else:
                    mod.fit_mod(**kwargs)
            except ValueError as msg:
                print(msg)
                print("-"*40)
                print("Skipping {}".format(name))

        self.__set_fit_models_to_spectrum()
        self.__generate_group_fit_table()


    def init_fitting(self, model, guess):
        tmp = {}
        for id, spec in self.spectra.group.items():
            if spec.signal.pass_snr:
                fit = FitSpectrum(spec.signal, model, **guess[id])
                tmp.update({id: fit})
        self.models = tmp


    def set_const(self, pname, value, id=None):
        if id is None:
            for mod in self.models.values():
                mod.set_const(pname, value)
        else:
            if id in self.models.keys():
                self.models[id].set_const(pname, value)

    def set_bounds(self, pname, min=None, max=None):
        for mod in self.models.values():
            mod.set_bounds(pname, min, max)

    def reset(self, name='all'):
        if name.upper() == 'ALL':
            for mod in self.models.values():
                mod.reset()
        else:
            if name.upper() in self.models.keys():
                self.model[name].reset()
            else:
                print('WARNING: {} not in available channels.'.format(
                    name.upper()))

    def quick_vis(self):
        l = self.__len__()
        l = int((l/2 + (l%2)/2)/2)
        fig, axes = plt.subplots(l, l, figsize=(14,12))
        axes=axes.flatten()
        for ax, mod in zip(axes, self.models.values()):
            if mod.result is None or not mod.pass_fitting:
                ax.set_title("Fitting Failed for {}".format(mod.sig.id))
            else:
                ax = mod.quick_vis(ax)


    @staticmethod
    def create_simple_guess(spectra):
        guess = {}
        for id, spec in spectra.group.items():
            inds = np.where((spec.signal.freq>=spec.signal.ubfreqs[0]) & (
                 spec.signal.freq<=spec.signal.ubfreqs[1]))
            llpsp = np.log10(spec.signal.amp[inds].max())
            fc = spec.signal.freq[inds][spec.signal.amp[inds].argmax()]
            guess.update({ id : {'llpsp':llpsp, 'fc': fc, 'ts': 0.01}})
        return guess

    @staticmethod
    def write_flatfile(path, fits):
        os.makedirs(os.path.join(*path.split("/")[:-1]), exist_ok=True)
        fits.table.to_csv(path, index=False)

    @staticmethod
    def read_flatfile(path):
        return pd.read_csv(path)

    def __check_wm(self, wm):
        if wm not in ['log', 'none']:
            print('WARNING: did not recognise weight method {}.'.format(
                    weight_method))
            print('Setting to none...')
            wm = 'none'
        return wm


    def __generate_group_fit_table(self):
        ds = [m.meta for m in self.models.values()]
        df1 = pd.DataFrame([])
        for i, d in enumerate(ds):
            df1 = pd.concat([df1, pd.DataFrame(d, index=[i])],
                ignore_index=True, sort=False)
        self.table = df1


    def __set_fit_models_to_spectrum(self):
        for id, mod in self.models.items():
            tmp = self.spectra.get_spectra(id)
            tmp.signal.set_model(mod)

    def __check_spectra(self, spectra):
        if type(spectra) is not type(sp.Spectra()):
            raise ValueError(
                "Must be a signal object not {}".format(type(signal)))
        else:
            return True
