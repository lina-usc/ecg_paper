import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import mne
import heartpy as hp
from scipy.stats import lognorm
from scipy import interpolate

from gym import Env
from gym.spaces import Box, Dict


path_timelog_format = ("Create_Segments/all_infants_timelogs/" +
                       "{subject}_{age}.csv")
datavyu_format = ("Generated Files_{kind}_03092022_Datavyu_ALLOnly_AI/" +
                  "Stimuli/{subject}_{age}_stimulus.csv")


def get_outliers(vals, k=3):
    q1, q2, q3 = np.percentile(vals, [25, 50, 75])
    return (vals < q2 - k * (q3 - q1)) | (vals > q2 + k * (q3 - q1))


def process_ecg_segment(file_name, debug=False):
    raw = mne.io.read_raw_edf(file_name, verbose=False)
    sfreq = raw.info["sfreq"]

    rec_ecg = raw.get_data().squeeze()

    return process_ecg_data_segment(rec_ecg, sfreq, debug=debug)


def process_ecg_data_segment(rec_ecg, sfreq, margin=250, debug=False,
                             resample=None, **kwargs):
    if len(rec_ecg) == 0:
        return

    if resample is not None and resample != sfreq:
        rec_ecg = mne.filter.resample(rec_ecg, up=resample/sfreq)
        sfreq = resample

    try:
        wd, m = hp.process(rec_ecg, sfreq, **kwargs)
    except hp.exceptions.BadSignalWarning:
        return

    beats = []
    peaks, peaks_y = np.array([[peak, peak_y]
                               for peak, peak_y
                               in zip(wd["peaklist"], wd["ybeat"])
                               if peak not in wd["removed_beats"]]).T
    peaks = peaks.astype(int)

    p1s, p2s, p3s = peaks[:-2], peaks[1:-1], peaks[2:]
    for p1, p2, p3 in zip(p1s, p2s, p3s):
        x = np.arange(p1, p3 + 1)
        y = rec_ecg[p1:(p3 + 1)]
        f = interpolate.interp1d(x, y)

        xnew = np.concatenate((np.linspace(p1, p2, margin),
                               np.linspace(p2, p3, margin)))
        ynew = f(xnew)  # use interpolation function returned by `interp1d`
        beats.append(ynew)

    beats = np.array(beats)

    if len(beats) == 0:
        return
    residuals = np.trapz((beats - beats.mean(0)) ** 2, axis=1)
    outliers = get_outliers(residuals, k=6)

    nb_samples = np.median((p3s - p1s)[~outliers])

    # Flagging as outliers P2 to close to the borders
    outliers |= ((p2s - nb_samples // 2).astype(int) < 0)
    outliers |= ((p2s + nb_samples // 2).astype(int) >= len(rec_ecg))

    additional_removed_beats = np.array(peaks)[np.concatenate([[True],
                                                               outliers,
                                                               [True]])]
    additional_removed_beats_y = np.array(peaks_y)[np.concatenate([[True],
                                                                   outliers,
                                                                   [True]])]

    clean_beats = beats[~outliers, :]

    raw_beats = np.array([rec_ecg[int(p2 - nb_samples // 2):
                                  int(p2 + nb_samples // 2)]
                          for p2 in p2s[~outliers]])

    raw_t = np.arange(-int(nb_samples // 2), int(nb_samples // 2)) / sfreq

    if clean_beats.shape[0] < 20:
        return

    wd_copy = wd.copy()

    wd_copy["removed_beats"] = np.concatenate([wd["removed_beats"],
                                               additional_removed_beats])
    wd_copy["removed_beats_y"] = np.concatenate([wd["removed_beats_y"],
                                                 additional_removed_beats_y])

    clean_mean_beat = np.median(clean_beats, 0)

    signal = np.trapz(clean_mean_beat ** 2)
    noise = np.trapz((clean_beats - clean_mean_beat) ** 2, axis=1)

    if debug:
        plt.figure()
        hp.plotter(wd, m, figsize=(20, 4))
        plt.xlim(0, 30)

        plt.figure()
        plt.plot(residuals, ".")
        plt.plot(np.arange(len(residuals))[outliers],
                 residuals[outliers], ".", color="r")

        plt.figure()
        hp.plotter(wd_copy, m, figsize=(20, 4))
        plt.xlim(0, 30)

        plt.figure()
        sns.heatmap(clean_beats)

        plt.figure()
        plt.plot(clean_beats.T, alpha=0.1, color='k')
        plt.plot(clean_mean_beat, color="r")

    return {"SNR": np.mean(10 * np.log10(signal / noise)),
            "mean_beat": clean_mean_beat,
            "nb_valid_beats": clean_beats.shape[0],
            "nb_invalid_beats": np.sum(outliers),
            # "file_parts": file_name.name.replace(".edf", "").split("_"),
            "wd": wd_copy,
            "clean_beats": clean_beats,
            "raw_beats": raw_beats,
            "raw_t": raw_t,
            "rel_p1": p2s[~outliers] - p1s[~outliers],
            "rel_p3": p3s[~outliers] - p2s[~outliers],
            "sfreq": sfreq}


def get_log_times(subject, age,
                  path_timelog_format=path_timelog_format,
                  datavyu_format=datavyu_format):

    # Look first for datavyu times
    rows = []
    for kind in ["OIX", "PIX"]:
        stim_path = Path(datavyu_format.format(kind=kind,
                                               subject=subject,
                                               age=age))
        if stim_path.exists():
            csv_file = pd.read_csv(stim_path).dropna()
            csv_file.columns = ["start", "end", "stimulus"]
            csv_file = csv_file[csv_file.stimulus != "END"]
            rows.append({"start": csv_file.start.min() / 60.0,
                         "end": csv_file.end.max() / 60.0,
                         "condition": kind})
    if len(rows):
        return pd.DataFrame(rows)

    # if datavyu times are not available, look for old time logs
    path_timelog = Path(path_timelog_format.format(subject=subject, age=age))
    if path_timelog.exists():
        csv_file = pd.read_csv(path_timelog).dropna()
        csv_file.columns = ["visit", "segment", "condition", "start", "end"]
        csv_file = csv_file[csv_file.end > csv_file.start]
        return csv_file

    # No segment logs available
    return None


def get_segments(path_edf, **kwargs):

    subject, age = path_edf.name.replace(".edf", "").split("_")
    log_df = get_log_times(subject, age, **kwargs)
    if log_df is None:
        return None

    edf_raw = mne.io.read_raw_edf(path_edf, preload=True)
    sfreq = edf_raw.info["sfreq"]

    edf_raw = edf_raw.notch_filter(np.arange(60, sfreq/2.0, 60))
    edf_raw = edf_raw.filter(1, sfreq/4.0)

    try:
        starts = (log_df.start * 60 * sfreq).astype(int)
        stops = (log_df.end * 60 * sfreq).astype(int)
    except:
        print(log_df)
        raise

    # Reading each row start stop in excel file (timelogs)
    segments = []
    for start, stop, condition in zip(starts, stops, log_df.condition.values):
        if stop > len(edf_raw.times):
            warnings.warn(f"Condition {condition} for file {path_edf.name}"
                          f" stop at sample {stop} while the recording "
                          f"contains only {len(edf_raw.times)} samples.")
        segment = edf_raw.get_data("ECG0", start, stop).squeeze()
        if segment is not None and len(segment):
            segments.append(segment)
    return segments, log_df.condition.values, sfreq


def params_dict_to_array(params):
    return np.array([list(ln.values()) for ln in params.values()]).ravel()


def params_array_to_dict(params, log_labels=None):
    if log_labels is None:
        return {f"ln{i}": dict(zip(["mu", "sigma", "t0", "D"], row))
                for i, row in enumerate(params.reshape(len(params) // 4, 4))}

    return {log_label: dict(zip(["mu", "sigma", "t0", "D"], row))
            for row, log_label
            in zip(params.reshape(len(params) // 4, 4), log_labels)}


def compute_snr(params, t, mean_beat):
    sim_beat = np.sum(np.array([lognpdf(t, **ln)
                                for ln in params.values()]),
                      axis=0)
    return 10 * np.log10(np.trapz(mean_beat ** 2) /
                         np.trapz((mean_beat - sim_beat) ** 2))


class SigmaLog:

    def __init__(self, params):
        self.params = params

    def __add__(self, other):
        return self.__op__(other, "__add__")

    def __radd__(self, other):
        return self.__op__(other, "__radd__")

    def __rsub__(self, other):
        return self.__op__(other, "__rsub__")

    def __sub__(self, other):
        return self.__op__(other, "__sub__")

    def __rmul__(self, other):
        return self.__op__(other, "__rmul__")

    def __mul__(self, other):
        return self.__op__(other, "__mul__")

    def __truediv__(self, other):
        return self.__op__(other, "__truediv__")

    def __iter__(self, *args):
        return self.params.__iter__(*args)

    def __next__(self, *args):
        return self.params.__next__(*args)

    def __getitem__(self, i):
        return self.params[i]

    def __str__(self):
        return str(self.params)

    def __repr__(self):
        return str(self.params)

    def __op__(self, other, fct):

        if isinstance(other, (SigmaLog, dict)):
            ret = {ln: {p: getattr(float(self.params[ln][p]),
                                   fct)(other.params[ln][p])
                        for p in self.params[ln]}
                   for ln in self.params}

        else:
            ret = {ln: {p: getattr(float(self.params[ln][p]), fct)(other)
                        for p in self.params[ln]}
                   for ln in self.params}

        return SigmaLog(ret)

    def abs(self):
        ret = {ln: {p: np.abs(self.params[ln][p])
                    for p in self.params[ln]}
               for ln in self.params}
        return SigmaLog(ret)

    def to_array(self):
        return params_dict_to_array(self.params)


def lognpdf(t, mu=0.0, sigma=1.0, t0=0.0, D=1.0):
    return D * lognorm(s=sigma, loc=t0, scale=np.exp(mu)).pdf(t)


def lognpdf3(t, mu=0.0, sigma=1.0, t0=0.0):
    return lognpdf(t, mu, sigma, t0)


def plot_fit_heart_beat(params, t, mean_beat, lower, upper,
                        show_logs=True, show_orig=True,
                        show_recon=True, show_bounds=True,
                        ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    sim_beat = np.sum(np.array([lognpdf(t, **ln) for ln in params.values()]),
                      axis=0)
    if show_orig:
        ax.plot(t, mean_beat)

    if show_recon:
        ax.plot(t, sim_beat)
    if show_logs:
        for ln in params.values():
            ax.plot(t, lognpdf(t, **ln), alpha=0.2)

    if show_bounds:
        ubs, lbs = np.array([get_bounds(t, param_lower, param_upper)
                             for param_lower, param_upper
                             in zip(lower.values(), upper.values())]
                            ).transpose([1, 0, 2])
        ub = np.array(ubs).sum(0)
        lb = np.array(lbs).sum(0)

        ax.plot(t, ub, color="r", linestyle='dashed', alpha=0.5)
        ax.plot(t, lb, color="b", linestyle='dashed', alpha=0.5)
        ax.plot(t, mean_beat)

    snr = 10 * np.log10(np.trapz(mean_beat ** 2) /
                        np.trapz((mean_beat - sim_beat) ** 2))
    ax.set_title(f"SNR: {np.round(snr, 2)}dB")


def get_upper_bound(time, param_lower, param_upper):
    """
     Eq 13: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6183528
    """
    upper_bound = np.zeros_like(time)

    masks = [time > param_lower["t0"],
             time > param_lower["t0"] + np.exp(param_lower["mu"] -
                                               param_upper["sigma"]),
             time > param_lower["t0"] + np.exp(param_lower["mu"] -
                                               param_lower["sigma"]),
             time > param_lower["t0"] + np.exp(param_lower["mu"] -
                                               param_lower["sigma"] ** 2),
             time > param_upper["t0"] + np.exp(param_lower["mu"] -
                                               param_lower["sigma"] ** 2),
             time > param_upper["t0"] + np.exp(param_lower["mu"]),
             time > param_upper["t0"] + np.exp(param_upper["mu"]),
             time > param_upper["t0"] + np.exp(param_upper["mu"] +
                                               param_lower["sigma"]),
             time > param_upper["t0"] + np.exp(param_upper["mu"] +
                                               param_upper["sigma"]),
             np.array(False)]
    masks = [m1 & ~m2 for m1, m2 in zip(masks[:-1], masks[1:])]

    upper_bound[masks[0]] = lognpdf3(time[masks[0]],
                                     t0=param_lower["t0"],
                                     mu=param_lower["mu"],
                                     sigma=param_upper["sigma"])
    sigma = param_lower["mu"] - np.log(time[masks[1]] - param_lower["t0"])
    upper_bound[masks[1]] = lognpdf3(time[masks[1]],
                                     t0=param_lower["t0"],
                                     mu=param_lower["mu"],
                                     sigma=sigma)
    upper_bound[masks[2]] = lognpdf3(time[masks[2]],
                                     t0=param_lower["t0"],
                                     mu=param_lower["mu"],
                                     sigma=param_lower["sigma"])
    upper_bound[masks[3]] = np.ones_like(time[masks[3]])
    upper_bound[masks[3]] *= lognpdf3(np.exp(param_lower["mu"] -
                                             param_lower["sigma"] ** 2),
                                      t0=0,
                                      mu=param_lower["mu"],
                                      sigma=param_lower["sigma"])
    upper_bound[masks[4]] = lognpdf3(time[masks[4]],
                                     t0=param_upper["t0"],
                                     mu=param_lower["mu"],
                                     sigma=param_lower["sigma"])
    upper_bound[masks[5]] = lognpdf3(time[masks[5]],
                                     t0=param_upper["t0"],
                                     mu=np.log(time[masks[5]] -
                                               param_upper["t0"]),
                                     sigma=param_lower["sigma"])
    upper_bound[masks[6]] = lognpdf3(time[masks[6]],
                                     t0=param_upper["t0"],
                                     mu=param_upper["mu"],
                                     sigma=param_lower["sigma"])
    sigma = np.log(time[masks[7]] - param_upper["t0"]) - param_upper["mu"]
    upper_bound[masks[7]] = lognpdf3(time[masks[7]],
                                     t0=param_upper["t0"],
                                     mu=param_upper["mu"],
                                     sigma=sigma)
    upper_bound[masks[8]] = lognpdf3(time[masks[8]],
                                     t0=param_upper["t0"],
                                     mu=param_upper["mu"],
                                     sigma=param_upper["sigma"])
    return upper_bound


def get_lower_bound(time, param_lower, param_upper):
    """
     Eq 22: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6183528
    """

    t0s = np.array([param_upper["t0"]] * 2 + [param_lower["t0"]] * 3)[:, None]
    mus = np.array([param_upper["mu"]] * 3 + [param_lower["mu"]] * 2)[:, None]
    sigmas = np.array([param_lower["sigma"]] +
                      [param_upper["sigma"]] * 3 +
                      [param_lower["sigma"]])[:, None]
    return lognpdf3(time, t0=t0s, mu=mus, sigma=sigmas).min(0)


def get_bounds(time, param_lower, param_upper):
    assert (np.all([param_lower[param] < param_upper[param]
                    for param in param_lower]))
    lower_bound = get_lower_bound(time, param_lower, param_upper)
    upper_bound = get_upper_bound(time, param_lower, param_upper)

    return (param_upper["D"] * upper_bound
            if param_upper["D"] > 0
            else param_upper["D"] * lower_bound,
            param_lower["D"] * lower_bound
            if param_lower["D"] > 0
            else param_lower["D"] * upper_bound)


# SigmaECGEnv(row["raw_beats"], lower_bound, upper_bound,
#                            row.sfreq, row.rel_p1, row.rel_p3)

def get_p2(params):
    return params["t0"] + np.exp(params["mu"]) * np.exp(-params["sigma"] ** 2)


def get_template_ecg():
    ecg_beat_template = {"P": dict(mu=-2.0, sigma=0.1,  t0=-0.24,  D=3e-6),
                         "Q": dict(mu=-3.0, sigma=0.4,  t0=-0.08,  D=-50e-6),
                         "R": dict(mu=-3.0, sigma=0.25, t0=-0.045, D=80e-6),
                         "S": dict(mu=-3.5, sigma=0.4,  t0=0.015, D=-10e-6),
                         "T": dict(mu=-1.0, sigma=0.4,  t0=0.15,  D=150e-6),
                         "U": dict(mu=-1.0, sigma=0.23, t0=0.22,  D=-120e-6),
                         }
    ecg_beat_template = SigmaLog(ecg_beat_template)

    delta = 0.2 * ecg_beat_template.abs()
    lower_bound = ecg_beat_template - delta
    upper_bound = ecg_beat_template + delta

    for comp in upper_bound:
        if lower_bound[comp]["D"] > 0:
            lower_bound[comp]["D"] = 0
        else:
            upper_bound[comp]["D"] = 0

    upper_bound["T"]["D"] += delta[comp]["D"]

    return ecg_beat_template, lower_bound, upper_bound


class SigmaECGEnv(Env):
    max_iter = 1000
    max_no_progress = 100

    min_frac = 0.3
    max_frac = 0.7
    margin = 1.0  # in seconds
    sfreq = 250

    imin = 175
    imax = 425
    t = np.arange(-margin * min_frac, margin * max_frac, 1 / sfreq)

    def __init__(self, segment_df, lower, upper, prototype):

        self.segment_df = segment_df

        self.lower = lower
        self.upper = upper
        self.ref = params_dict_to_array(prototype.params)

        v_max = segment_df.mean_beat.apply(np.max).max()
        v_min = segment_df.mean_beat.apply(np.min).min()

        shape = params_dict_to_array(self.upper.params).shape
        self.action_space = Box(-0.01, 0.01, shape=shape)
        spaces = {
            "error": Box(low=np.ones_like(self.t) * 1.5 * v_min,
                         high=np.ones_like(self.t) * 1.5 * v_max,
                         dtype=np.float64),
            "estimate": Box(self.lower.to_array(), self.upper.to_array(),
                            dtype=np.float64),
        }
        self.observation_space = Dict(spaces=spaces)
        self.reset()

    def action_to_beat(self, action):
        updated_estimate = self.estimate + (self.ref * action)

        # Ensuring to keep the t2 order.
        for i, (p21, p22) in enumerate(zip("PQRST", "QRSTU")):
            t21 = get_p2(params_array_to_dict(updated_estimate, "PQRSTU")[p21])
            t22 = get_p2(params_array_to_dict(updated_estimate, "PQRSTU")[p22])

            if t22 < t21:
                action[4 * i + 0] = min(action[4 * i + 0], 0)
                action[4 * i + 1] = max(action[4 * i + 1], 0)
                action[4 * i + 2] = min(action[4 * i + 2], 0)

                action[4 * (i + 1) + 0] = max(action[4 * (i + 1) + 0], 0)
                action[4 * (i + 1) + 1] = min(action[4 * (i + 1) + 1], 0)
                action[4 * (i + 1) + 2] = max(action[4 * (i + 1) + 2], 0)

                updated_estimate = self.estimate + (self.ref * action)

        # Ensuring that no parameters go out of border
        lower_params = params_dict_to_array(self.lower.params)
        updated_estimate = np.array([updated_estimate, lower_params]).max(0)
        upper_params = params_dict_to_array(self.upper.params)
        self.estimate = np.array([updated_estimate, upper_params]).min(0)

        params = self.estimate.reshape(len(self.estimate) // 4, 4)
        return np.sum(lognpdf(self.t, mu=params.T[0, :, None],
                              sigma=params.T[1, :, None],
                              t0=params.T[2, :, None],
                              D=params.T[3, :, None]), axis=0)

    def step(self, action):
        self.action = action
        self.episode_length -= 1

        sim_beat = self.action_to_beat(action)
        self.state = {
            "error": np.array(self.target_beat - sim_beat),
            "estimate": self.estimate,
        }

        signal = np.trapz(self.target_beat ** 2)
        noise = np.trapz((self.target_beat - sim_beat) ** 2)
        snr = 10 * np.log10(signal / noise)
        snr = -100 if np.isnan(snr) or np.isinf(snr) or snr < -100 else snr
        if np.isnan(self.last_snr):
            reward = snr
            self.best_snr = snr
            self.best_solution = self.estimate.copy()
            self.best_step = self.episode_length
        else:
            reward = snr - self.last_snr
            if snr > self.best_snr:
                self.best_snr = snr
                self.best_solution = self.estimate.copy()
                self.best_step = self.episode_length

        self.last_snr = snr
        done = ((self.episode_length <= 0) or
                (self.episode_length <= self.best_step - self.max_no_progress))
        info = {}

        return self.state, reward, done, info

    def render(self):
        print(f"best_snr: {self.best_snr};    subject: {self.row.subject};"
              f"     age: {self.row.age};    condition: {self.row.condition};"
              f"     beat_no: {self.selected_beat_ind}")

    def set_target_beat(self, signal):
        return signal[int(self.margin * self.sfreq * (1 - self.min_frac)):
                      int(self.margin * self.sfreq * (1 + self.max_frac))]

    def reset(self, random_state=1):
        return self.reset_set(random_state=random_state)

    def reset_set(self, row=None, selected_beat_ind=None, mean=False,
                  random_state=1):
        if row is None:
            self.row = self.segment_df.sample(random_state=random_state)
            self.row = self.row.squeeze()
        else:
            self.row = row

        if mean:
            self.target_beat = self.set_target_beat(self.row.mean_beat)
            self.selected_beat_ind = "mean"
        else:
            if selected_beat_ind is None:
                indices = np.arange(len(self.row.rel_p1))
                selected_beat_ind = np.random.choice(indices, 1)[0]

            self.selected_beat_ind = selected_beat_ind

            target_beat = self.row.clean_beats[self.selected_beat_ind]
            self.target_beat = self.set_target_beat(target_beat)

        self.last_snr = np.nan
        self.best_snr = np.nan
        self.best_solution = None
        self.estimate = self.ref
        action = np.zeros_like(params_dict_to_array(self.upper.params))
        sim_beat = self.action_to_beat(action)

        self.state = {
            "error": np.array(self.target_beat - sim_beat),
            "estimate": self.estimate,
        }
        self.episode_length = self.max_iter
        return self.state
