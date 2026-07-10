#! /usr/bin/env python
"""Library of useful functions for statistical analysis of morphometrics data

If called as a script, this will assemble an Experiment object based on the config.yml file
Usage: python morphometrics_stats.py config.yml experiment_name"""

__author__ = "Benjamin Barad"
__email__ = "benjamin.barad@gmail.com"
__license__ = "GPLv3"

import copy
import glob
import os
import pickle
import warnings

import click
import csv
import yaml

from .config_utils import load_config
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from textwrap import wrap
import scipy.stats as st

SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 10.5

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import numpy as np
import pandas as pd
import scipy.stats as stats

purple = [.5, 0, .5, 0.9]
purple_light = [.5,0,.5, 0.3]
green = [0, 1, 0, 0.9]
green_light = [0,1,0,.3]
blue = [0,.3,1, .9]
blue_light = [0,.3,1,.3]

my_cmap = copy.copy(matplotlib.colormaps['viridis']) # copy the default cmap
my_cmap.set_bad(my_cmap.colors[0])

colors = [purple,green, blue]*4
colors_light = [purple_light, green_light, blue_light]*4
# colors = ["#5DA5DA", "#FAA43A", "#60BD68", "#F17CB0", "#B276B2", "#DECF3F", "#F15854", "#4D4D4D"] # "#B2912F"
# face_alphas = [0.3]*len(colors)
# colors_light = list(zip(colors, face_alphas))


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

class Experiment():
    """Experiments are containers for a series of tomograms and their pandas dataframes
    
    Access pattern: experiment[tomoname][label]."""
    def __init__(self, name):
        self.name = name
        self.tomograms = {}
        self.tomogram_names = set()
    

    def add_tomograms(self, names, surface_bases, folder, file_extension):
        """Add tomograms to the experiment. Names must be the basenames used by pycurv.
        
        Args:
            names (list): list of tomogram prefixes
            surface_bases (list): list of label names
            folder (str): path to the folder containing surfaces
            file_extension (str): file extension of the surface files"""
        for name in names:
            # print(name)
            surfs = []
            fns = []
            for surface_base in surface_bases:
                filename = folder+name+"_"+surface_base+file_extension
                print(filename)
                if os.path.isfile(filename):
                    surfs.append(surface_base)
                    fns.append(filename)
            self.tomograms[name] = Tomogram(name, surface_bases, fns)

    def __getitem__(self, name):
        return self.tomograms[name]
    
    def __setitem__(self, name, tomo):
        self.tomogram_names.add(name)
        self.tomograms[name] = tomo

class Tomogram():
    """A tomogram class, which can have a number of different dataframes associated with it.
    Each dataframe is associated with a differen segmented feature, such as ER or IMM.
"""
    
    def __init__(self, name, groupnames, csvnames):
        self.name = name
        print(self.name)
        self.dataframes = {}
        self.dataframe_names = set()
        for index, csvname in enumerate(csvnames):
            print(groupnames[index])
            self.add_dataframe(groupnames[index], pd.read_csv(csvname))
    
    def add_dataframe(self, name, dataframe):
        self.dataframes[name] = dataframe
        self.dataframe_names.add(name)
    
    def has_key(self, key):
        return key in self.dataframe_names
    
    def __getitem__(self, key):
        if not key in self.dataframe_names:
            raise KeyError("Tomogram does not have dataframe {}".format(key))
        return self.dataframes[key]
    
    def __setitem__(self, key, value):
        self.dataframe_names.add(key)
        self.dataframes[key] = value

def weighted_median(values, weights):
    """compute the weighted median of values list. The 
    weighted median is computed as follows:
    1- sort both lists (values and weights) based on values.
    2- select the 0.5 point from the weights and return the corresponding values as results
    e.g. values = [1, 3, 0] and weights=[0.1, 0.3, 0.6] assuming weights are probabilities.
    sorted values = [0, 1, 3] and corresponding sorted weights = [0.6, 0.1, 0.3] the 0.5 point on
    weight corresponds to the first item which is 0. so the weighted median is 0.
    
    Function provided by Max Ghenis on stack overflow, CC-BY-SA
    
    Args:
        values (array-like): list of values
        weights (array-like): list of weights"""

    #convert the weights into probabilities
    sum_weights = sum(weights)
    weights = np.array([(w*1.0)/sum_weights for w in weights])
    #sort values and weights based on values
    values = np.array(values)
    sorted_indices = np.argsort(values)
    values_sorted  = values[sorted_indices]
    weights_sorted = weights[sorted_indices]
    #select the median point
    it = np.nditer(weights_sorted, flags=['f_index'])
    accumulative_probability = 0
    median_index = -1
    while not it.finished:
        accumulative_probability += it[0]
        if accumulative_probability > 0.5:
            median_index = it.index
            return values_sorted[median_index]
        elif accumulative_probability == 0.5:
            median_index = it.index
            it.iternext()
            next_median_index = it.index
            return np.mean(values_sorted[[median_index, next_median_index]])
        it.iternext()

    return values_sorted[median_index]

def weighted_histogram_peak(values, weights, bins, bin_range):
    '''compute the peak of a histogram for a list of values.
    
    Args:
        values (array-like): list of values
        weights (array-like): list of weights
        bins (array-like): list of bins
        bin_range (tuple): (min, max) of the histogram
    '''
    hist, bin_edges = np.histogram(values, bins=bins, range=bin_range, weights=weights, density=True)
    i = np.argmax(hist)
    bin_vals = (bin_edges[i], bin_edges[i+1])
    return(np.mean(bin_vals))

def ks_statistics(datasets, areas, ns, condition_names, morph_names, basename, filename, rads=[9,12,15]):
    """Compute the two-sided KS test for a pair of sets of mitochondria, with a variety of different assumptions about the size of the independent variables.
    
    Args:
        datasets (list): list of datasets to compare
        areas (list): total area for each dataset
        ns (list): number of mitochondria for each dataset
        condition_names (list): condition names for each dataset
        morph_names (list): morphology names for each dataset
        basename (str): type of distribution to test
        filename (str): name of the file to save the results to
        rad (int): radius of the smallest feature for estimation of independent features. Default to 8 based on intracrista spacing at bend sites.
        """
    area_adjusters = [np.pi*rad**2 for rad in rads]
    with open(filename, "w") as f:
        radnames = ",".join([f"prad({str(r)})" for r in rads])
        f.write(f"Base Experiment,Stat Type,Sample A Condition,Sample A Morph,Sample B Condition,Sample B Morph,KS_stat,pbase,pmito,{radnames}\n")
        for i, set_a in enumerate(datasets):
            for j, set_b in enumerate(datasets):
                if j<=i: 
                    continue
                inds_n_1 = [areas[i]/area_adjuster for area_adjuster in area_adjusters]
                inds_n_2 = [areas[j]/area_adjuster for area_adjuster in area_adjusters]
                inds_en = [round(np.mean([inds_n_1[i], inds_n_2[i]])) for i in range(len(rads))]
                en = round(np.mean([ns[i], ns[j]]))
                ks, p = stats.ks_2samp(set_a, set_b)
                # print(ks,p)
                p_en = stats.distributions.kstwo.sf(ks, en)
                # print(en, p_en)
                p_rad = ",".join([str(stats.distributions.kstwo.sf(ks, ind_en)) for ind_en in inds_en])
                # print(ind_en, p_rad)
                f.write(f"{basename},KS,{condition_names[i]},{morph_names[i]},{condition_names[j]},{morph_names[j]},{ks},{p},{p_en},{p_rad}\n")
                

def significance_stars(p, ns="ns", na="n/a"):
    """Star notation for a p-value, using this project's thresholds.

    **** p<0.001, *** p<0.005, ** p<0.01, * p<0.05.

    `ns` is returned for a finite, non-significant p; `na` for a missing/undefined p
    (e.g. a degenerate comparison). Both are parameterised so callers with a different
    legacy convention can reproduce it exactly (e.g. statistics() uses ns=na=" ").
    """
    if p is None or not np.isfinite(p):
        return na
    if p < 0.001:
        return "****"
    if p < 0.005:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ns


def pairwise_tests(datasets, labels, include_ks=True, ns="ns", na="n/a"):
    """Pairwise significance tests between every pair of datasets.

    Operates on whatever you pass as `datasets` — in this codebase that is one
    *summary statistic per tomogram* (e.g. a weighted_histogram_peak), so the tests
    compare classes at the **per-tomogram summary level**, treating the tomogram as
    the independent unit.

    Always computes Mann-Whitney U (rank test, no normality assumption) and Welch's
    t-test (unequal-variance means). When `include_ks` is True, also computes a
    two-sample Kolmogorov-Smirnov test **on those same summary values** — this is a
    paired summary-statistic KS, NOT the pooled triangle-distribution KS with the
    effective-n correction in `ks_statistics()`.

    Degenerate comparisons (identical or near-identical groups) return NaN -> `na`
    rather than raising, and scipy's precision-loss warning is suppressed.

    Returns a list of dicts, one per pair (i<j), ordered by dataset index.
    """
    rows = []
    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            a = np.asarray(datasets[i], dtype=float)
            b = np.asarray(datasets[j], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    u, p_u = np.nan, np.nan
                try:
                    t, p_t = stats.ttest_ind(a, b, equal_var=False)   # Welch
                except (ValueError, ZeroDivisionError):
                    t, p_t = np.nan, np.nan
                row = {
                    "class_a": labels[i], "class_b": labels[j],
                    "n_a": len(a), "n_b": len(b),
                    "mean_a": np.mean(a), "mean_b": np.mean(b),
                    "ci95_a": st.sem(a) * 1.96, "ci95_b": st.sem(b) * 1.96,
                    "mwu_U": u, "mwu_p": p_u, "mwu_stars": significance_stars(p_u, ns, na),
                    "ttest_t": t, "ttest_p": p_t, "ttest_stars": significance_stars(p_t, ns, na),
                }
                if include_ks:
                    try:
                        ks, p_ks = stats.ks_2samp(a, b)
                    except ValueError:
                        ks, p_ks = np.nan, np.nan
                    row.update(ks_stat=ks, ks_p=p_ks,
                               ks_stars=significance_stars(p_ks, ns, na))
            rows.append(row)
    return rows


def statistics(datasets, basename,  condition_names, morph_names, test_type="median", filename="test.csv", figsize=(5,4), ylabel="Peak Value", custom_colors=None, separator_after=None):
    """Generate mann-whitney U test and violin plots for a set of datasets.

    The pairwise CSV's KS columns (KS_summary_Stars / KS_summary_stat / P_KS_summary)
    compare the per-tomogram SUMMARY statistics you pass in — not pooled triangle
    distributions. For the distribution-level KS with the effective-n correction
    (pmito / prad), use ks_statistics().

    Args:
        datasets (list): list of datasets to compare
        basename (str): base name of the output files
        condition_names (list): list of condition names
        morph_names (list): list of morph names
        test_type (str): type of test to perform.
        filename (str): name of the output file
        figsize (tuple): size of the figure
        ylabel (str): label of the y axis
        custom_colors (list): optional list of colors for each violin. Defaults to module colors.
        separator_after (int): if set, draw a light grey dashed vertical line after this violin
            index (1-based), e.g. separator_after=2 draws a line between violins 2 and 3.
        """
    _colors = custom_colors if custom_colors is not None else colors
    raw_filename = filename[:-10]+"rawstats.csv"
    with open(raw_filename, 'w') as raw_file:
        raw_file.write(basename+f" - {test_type}\n")
        for index, val in enumerate(datasets):
            raw_file.write(f"{condition_names[index]} {morph_names[index]},"+",".join(map(str, val))+"\n")
    # Pairwise tests come from the shared helper; ns=na=" " reproduces this function's
    # legacy blank token. The KS here is the paired summary-statistic KS (see
    # KS_summary_* below and pairwise_tests' docstring), not the distribution-level
    # test in ks_statistics().
    results = {(r["class_a"], r["class_b"]): r
               for r in pairwise_tests(datasets, list(range(len(datasets))),
                                       include_ks=True, ns=" ", na=" ")}
    with open(filename, 'w') as file:
        file.write("Base Experiment,Stat Type,Utest_Stars,KS_summary_Stars,Sample A Condition,Sample A Morph,Sample A Mean,Sample A 95% Error,Sample B Condition,Sample B Morph,Sample B Mean,Sample B 95% Error,U,P_U,T,P_T,KS_summary_stat,P_KS_summary,n_A,n_B\n")
        for i, set_a in enumerate(datasets):
            for j, set_b in enumerate(datasets):
                if j<=i:
                    continue
                r = results[(i, j)]
                file.write(f"{basename},{test_type},{r['mwu_stars']},{r['ks_stars']},{condition_names[i]},{morph_names[i]},{r['mean_a']},{r['ci95_a']},{condition_names[j]},{morph_names[j]},{r['mean_b']},{r['ci95_b']},{r['mwu_U']},{r['mwu_p']},{r['ttest_t']},{r['ttest_p']},{r['ks_stat']},{r['ks_p']},{r['n_a']},{r['n_b']}\n")
    figure_filename = filename[:-3]+"svg"
    fig,ax=plt.subplots(figsize=figsize)
    ax.set_title(basename)
    parts = ax.violinplot(datasets, showmeans=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(_colors[i % len(_colors)][:3]) + [0.4])
        pc.set_edgecolor(_colors[i % len(_colors)])
    if separator_after is not None:
        ax.axvline(x=separator_after + 0.5, color='lightgrey', linestyle='--', linewidth=1.5, zorder=0)
    ax.set_xticks(range(1,len(datasets)+1))
    ax.set_xticklabels([condition_names[i]+"\n"+morph_names[i] for i in range(len(condition_names))])
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    fig.savefig(figure_filename)
    fig.savefig(figure_filename[:-3]+"png")


def violin(datasets, labels, title, ylabel, filename="violin.svg", figsize=(5, 4),
           custom_colors=None, show_points=True, annotations=None, separator_after=None):
    """Violin plot of one distribution per label, with the observations overlaid.

    Used to compare a per-surface summary statistic (one value per tomogram) across
    membrane classes: one violin per class, one dot per tomogram.

    Args:
        datasets (list): list of 1D array-likes, one per label.
        labels (list): x tick label for each dataset.
        title (str): plot title.
        ylabel (str): y axis label.
        filename (str): output .svg path; a matching .png is written alongside.
        figsize (tuple): figure size in inches.
        custom_colors (list): optional per-violin colors. Defaults to module colors.
        show_points (bool): overlay the individual observations (one dot each).
        annotations (list): optional [(i, j, text), ...] significance brackets drawn
            above the violins, connecting dataset i to dataset j (0-based) and
            labelled with `text` (e.g. "**" or "ns"). Narrow comparisons are drawn
            lowest so wider brackets stack above them.
        separator_after (int): if set, draw a light grey dashed vertical line after
            this violin index (1-based), e.g. separator_after=2 draws a line between
            violins 2 and 3.
    """
    _colors = custom_colors if custom_colors is not None else colors
    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(datasets, showmeans=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        color = _colors[i % len(_colors)]
        pc.set_facecolor(list(color[:3]) + [0.4])
        pc.set_edgecolor(color)
    for key in ('cmeans', 'cmaxes', 'cmins', 'cbars'):
        if key in parts:
            parts[key].set_color('dimgrey')
    if show_points:
        # Deterministic jitter so repeated runs give identical figures.
        rng = np.random.default_rng(0)
        for i, data in enumerate(datasets):
            data = np.asarray(data, dtype=float)
            x = np.full(len(data), i + 1, dtype=float)
            if len(data) > 1:
                x += rng.uniform(-0.06, 0.06, size=len(data))
            ax.plot(x, data, 'o', markersize=3, alpha=0.8, zorder=3,
                    color=list(_colors[i % len(_colors)][:3]))
    if annotations:
        arrays = [np.asarray(d, dtype=float) for d in datasets if len(d)]
        data_max = max(a.max() for a in arrays)
        data_min = min(a.min() for a in arrays)
        span = (data_max - data_min) or abs(data_max) or 1.0
        tick = span * 0.02
        # Narrow comparisons first, so wider brackets nest above them.
        ordered = sorted(annotations, key=lambda ann: abs(ann[1] - ann[0]))
        top = data_max
        for level, (i, j, text) in enumerate(ordered):
            y = data_max + span * 0.09 * (level + 1)
            x1, x2 = i + 1, j + 1
            ax.plot([x1, x1, x2, x2], [y, y + tick, y + tick, y],
                    lw=1.0, color='dimgrey', zorder=4)
            ax.text((x1 + x2) / 2, y + tick, text, ha='center', va='bottom',
                    fontsize=9, color='dimgrey')
            top = y + tick
        ax.set_ylim(top=top + span * 0.09)
    if separator_after is not None:
        ax.axvline(x=separator_after + 0.5, color='lightgrey', linestyle='--',
                   linewidth=1.5, zorder=0)
    ax.set_xticks(range(1, len(datasets) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title("\n".join(wrap(title, 45)))
    plt.tight_layout()
    fig.savefig(filename)
    fig.savefig(filename[:-3] + "png")
    plt.close(fig)
    return filename


def histogram(data, areas, labels, title, xlabel, filename="hist.svg", bins=50, range=None, figsize=(6,4), logx=False, vlines = True, legend=True, color_offset=0, custom_colors=None, custom_colors_light=None):
    """Construct an area-weighted histogram of the data.
    Args:
        data (array-like): list of arrays to be independently plotted.
        areas (array-like): list of area values for each data array.
        labels (array-like): list of labels for each data array.
        title (str): title of the plot.
        xlabel (str): x-axis label.
        filename (str): name of the output file.
        bins (int): number of bins.
        range (list): range of the x-axis.
        figsize (tuple): figure size.
        logx (bool): whether to use a logarithmic x-axis.
        vlines (bool): whether to plot vertical lines at the histogram peaks.
        legend (bool): whether to show a legend. Default True.
        color_offset (int): offset for the color of the histogram. Default 0. Used for some weird side cases
        custom_colors (list): optional list of colors overriding the module defaults.
        custom_colors_light (list): optional list of light colors overriding the module defaults.
        """
    _colors       = custom_colors       if custom_colors       is not None else colors
    _colors_light = custom_colors_light if custom_colors_light is not None else colors_light
    assert len(data)==len(areas)
    assert len(data)==len(labels)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if logx:
        bins = np.logspace(np.log10(range[0]),np.log10(range[1]), bins)
        ax.set_xscale("log")
    for index, value in enumerate(data):
        n,binset,_ = ax.hist(value, bins=bins, weights=areas[index], label=labels[index], ec=_colors[index],fc=_colors_light[index],histtype="stepfilled", density=True, range = range) #
        if vlines:
            delta = (binset[1]-binset[0])/2
            idx = n.argmax()
            ax.axvline(binset[idx]+delta, linestyle="--", color=_colors[index])
    ax.set_xlim(range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative Area")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if legend:
        ax.legend()
    # plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    fig.savefig(filename[:-3]+"notitle.png", bbox_inches='tight', dpi=300)
    ax.set_title(title, loc='left')
    fig.savefig(filename[:-3]+"png", bbox_inches='tight', dpi=300)
    return True

def twod_histogram(data1, data2, areas, data1_label, data2_label, title, bins=(50,50), range=None, filename="twod_hist.svg", figsize=(5,5), log=True):
    """Construct an area-weighted 2D histogram of the data.
    
    Args:
        data1 (list): list of values to for x axis.
        data2 (list): list of values to for y axis.
        areas (list): list of area values for weighting.
        data1_label (str): label for x axis.
        data2_label (str): label for y axis.
        title (str): title of the plot.
        bins (tuple): number of bins for x and y axis.
        range (list): range of the x and y axis. (format is [xmin, xmax, ymin, ymax])
        filename (str): name of the output file.
        figsize (tuple): figure size.
        log (bool): whether to use a logarithmic x-axis.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.hist2d(data1, data2, bins=bins, range=range, density=True, weights=areas, norm=matplotlib.colors.LogNorm(), cmap=my_cmap)
    ax.set_xlabel(data1_label)
    ax.set_ylabel(data2_label)
    if range:
        ax.set_xlim(range[0])
        ax.set_ylim(range[1])
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    fig.savefig(filename[:-3]+"png", bbox_inches='tight')
    return True

def scatter_regression(x_data_sets, y_data_sets, area_sets, labels,
                       xlabel, ylabel, title_prefix,
                       filename="scatter_regression.svg",
                       figsize=None, xlim=None, ylim=None,
                       show_stats=True, alpha=0.3, s=1):
    """Create scatter plots with area-weighted linear regression for multiple datasets.

    Creates a grid of subplots, one per condition, each showing a scatter plot
    with weighted linear regression line and statistics.

    Args:
        x_data_sets (list of lists): x values for each condition
        y_data_sets (list of lists): y values for each condition
        area_sets (list of lists): area weights for each condition (used for regression weighting)
        labels (list): labels for each condition
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title_prefix (str): prefix for subplot titles (label will be appended)
        filename (str): output filename
        figsize (tuple): figure size (default: auto-calculated based on number of conditions)
        xlim (tuple): x-axis limits (min, max)
        ylim (tuple): y-axis limits (min, max)
        show_stats (bool): whether to show regression statistics on plot
        alpha (float): transparency of scatter points
        s (float): size of scatter points
    """
    n_conditions = len(x_data_sets)

    # Calculate grid layout (prefer horizontal layout)
    if n_conditions <= 3:
        nrows, ncols = 1, n_conditions
    elif n_conditions == 4:
        nrows, ncols = 2, 2
    elif n_conditions == 5:
        nrows, ncols = 2, 3
    else:
        nrows = int(np.ceil(np.sqrt(n_conditions)))
        ncols = int(np.ceil(n_conditions / nrows))

    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_conditions == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, (x_data, y_data, areas, label) in enumerate(zip(x_data_sets, y_data_sets, area_sets, labels)):
        ax = axes[idx]

        if len(x_data) == 0 or len(y_data) == 0:
            ax.text(0.5, 0.5, f'No data\nfor {label}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            continue

        # Convert to numpy arrays
        x = np.array(x_data)
        y = np.array(y_data)
        w = np.array(areas)

        # Filter out NaN and inf values
        valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
        x = x[valid_mask]
        y = y[valid_mask]
        w = w[valid_mask]

        if len(x) == 0:
            ax.text(0.5, 0.5, f'No valid data\nfor {label}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            continue

        # Create scatter plot (area determines point size or use uniform small size)
        ax.scatter(x, y, s=s, alpha=alpha, color=colors[idx], rasterized=True)

        # Perform weighted linear regression
        # np.polyfit with weights performs weighted least squares
        coeffs = np.polyfit(x, y, 1, w=w)
        slope, intercept = coeffs

        # Calculate R² for weighted regression
        y_pred = slope * x + intercept
        ss_res = np.sum(w * (y - y_pred)**2)
        ss_tot = np.sum(w * (y - np.average(y, weights=w))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate p-value using unweighted regression (scipy doesn't support weighted)
        # This is a limitation but gives us a p-value estimate
        slope_uw, intercept_uw, r_value, p_value, std_err = stats.linregress(x, y)

        # Plot regression line
        x_line = np.array([x.min(), x.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=colors[idx], linewidth=2, alpha=0.8,
               label=f'y = {slope:.3f}x + {intercept:.2f}')

        # Add statistics text
        if show_stats:
            stats_text = f'$R^2$ = {r_squared:.3f}\np < {p_value:.1e}' if p_value < 0.001 else f'$R^2$ = {r_squared:.3f}\np = {p_value:.3f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=SMALL_SIZE,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title_prefix} - {label}')

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.grid(True, alpha=0.3)

    # Hide extra subplots if n_conditions < nrows*ncols
    for idx in range(n_conditions, nrows * ncols):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    fig.savefig(filename[:-3]+"png", bbox_inches='tight', dpi=300)
    plt.close(fig)
    return True

def barchart(bars, errors, labels, title, ylabel, filename="barchart.svg", figsize=(4,3), ymax=None, hline=None):
    """Construct a bar chart with error bars.
    
    Args:
        bars (list): list of bar heights.
        errors (list): list of error bars.
        labels (list): list of bar labels.
        title (str): title of the plot.
        ylabel (str): y-axis label.
        filename (str): name of the output file.
        figsize (tuple): figure size.
        ymax (float): maximum value of the y-axis.
        hline (float): Optional value of the horizontal line.
        """
    # print(len(bars))
    x = np.arange(len(bars))
    fig, ax = plt.subplots(figsize=figsize)
    barwidth=0.8
    ax.set_title(title)
    if hline:
        ax.axhline(hline, linestyle="--", alpha=0.5)
    ax.bar(x, bars, yerr=errors, tick_label=labels, width=barwidth, color ="0.7", edgecolor="0.3")
    ax.set_ylabel(ylabel)
    if ymax:
        ax.set_ylim(0,ymax)

    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    fig.savefig(filename[:-3]+"png", bbox_inches='tight')
    return True



def double_barchart(bars1, bars2, errors1, errors2, labels, title, ylabel, legends = ["Untreated", "Treated"], filename="double_barchart.svg", figsize=(4,3)):
    """Construct a side by side bar chart for 2 sets of data.

        Args:
            bars1 (list): list of values for the first set of bars.
            bars2 (list): list of values for the second set of bars.
            errors1 (list): list of errors for the first set of bars.
            errors2 (list): list of errors for the second set of bars.
            labels (list): list of labels for the bars.
            title (str): title of the plot.
            ylabel (str): label for the y-axis.
            legends (list): list of legends for the two datasets
            filename (str): name of the output file.
            figsize (tuple): figure size."""
    x = np.arange(len(bars1))
    barwidth = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.bar(x, bars1, yerr=errors1, width=barwidth, color = colors_light[0], edgecolor=colors[0], label=legends[0])
    ax.bar(x+barwidth, bars2, yerr=errors2, width=barwidth, color=colors_light[1], edgecolor=colors[1], label=legends[1])
    
    ax.set_ylabel(ylabel)
    ax.set_xticks(x+barwidth/2)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    fig.savefig(filename[:-3]+"png", bbox_inches='tight')

def bootstrap(sets, areas, conditions, morphologies, reps=1000, basename="bootstrap", filename="bootstrap.csv", bins=50, binrange=(0,100)):
    """Calculate the bootstrap statistics of a set of datasets.

    DEPRECATED for confidence intervals: this resamples individual triangles i.i.d.,
    which assumes they are independent. Triangles on a membrane are strongly spatially
    correlated, so the resulting intervals are far too narrow (coverage ~20% at nominal
    90% in simulation). Use surface_morphometrics.spatial_stats.cluster_t_interval
    (per-tomogram means -> t-interval; recommended) or cluster_bootstrap (whole-tomogram
    resample), which treat the tomogram as the unit of replication.
    """
    bootstrap_sets = []
    with open(filename, "w") as f:
        f.write("Experiment,Condition,5%,50%,95%,area,n_triangles\n")

        for index, set in enumerate(sets):
            set = np.array(set)
            area = np.array(areas[index])
            condition = conditions[index]
            morphology = morphologies[index]
            bootstraps = np.zeros(reps)

            for i in range(reps):
                randn = np.random.randint(0, len(set), len(set))
                randset = set[randn]
                randarea = area[randn]
                bootstraps[i] = weighted_histogram_peak(randset, randarea, bins=bins, bin_range=binrange)
            f.write(f"{basename},{condition} {morphology},{np.percentile(bootstraps, 5)},{np.percentile(bootstraps, 50)},{np.percentile(bootstraps, 95)},{sum(area)},{len(area)}\n")


    

@click.command()
@click.argument('configfile', type=click.Path(exists=True), required=True)
@click.argument('experimentname', required=True)
def assemble_experiment_pickle(configfile, experimentname):
    """Assemble a pickle file for an experiment from all the tomos in the config folder."""
    config = load_config(configfile, require=("seg_dir", "work_dir", "segmentation_values"))

    output_file = config["work_dir"] + experimentname+".pkl"

    input_files = glob.glob(config["seg_dir"] + "*.mrc")
    input_names = [os.path.basename(x)[:-4] for x in input_files]
    rh = config["curvature_measurements"]["radius_hit"]
    extension = ".AVV_rh{}.csv".format(rh)
    labels = list(config["segmentation_values"].keys())
    experiment = Experiment(experimentname)

    experiment.add_tomograms(input_names, labels, config["work_dir"], extension)
    with open(output_file, "wb") as f:
        pickle.dump(experiment, f)
    print(output_file)






if __name__ == "__main__":
    assemble_experiment_pickle()