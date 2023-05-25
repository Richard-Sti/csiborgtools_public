# Copyright (C) 2023 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from argparse import ArgumentParser
from os.path import join

import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function

import utils

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


@cache_to_disk(7)
def read_dist(simname, run, kind, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    return reader.build_dist(simname, run, kind, verbose=True)


@cache_to_disk(7)
def make_kl(simname, run, nsim, nobs, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    pdf = read_dist("quijote", run, "pdf", kwargs)
    return reader.kl_divergence(simname, run, nsim, pdf, nobs=nobs)


@cache_to_disk(7)
def make_ks(simname, run, nsim, nobs, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    cdf = read_dist("quijote", run, "cdf", kwargs)
    return reader.ks_significance(simname, run, nsim, cdf, nobs=nobs)


def plot_dist(run, kind, kwargs):
    """
    Plot the PDF/CDF of the nearest neighbour distance for Quijote and CSiBORG.
    """
    assert kind in ["pdf", "cdf"]
    print(f"Plotting the {kind}.", flush=True)
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    x = reader.bin_centres("neighbour")

    y_quijote = read_dist("quijote", run, kind, kwargs)
    y_csiborg = read_dist("csiborg", run, kind, kwargs)
    ncdf = y_csiborg.shape[0]

    with plt.style.context(utils.mplstyle):
        plt.figure()
        for i in range(ncdf):
            if i == 0:
                label1 = "Quijote"
                label2 = "CSiBORG"
            else:
                label1 = None
                label2 = None
            plt.plot(x, y_quijote[i], c="C0", label=label1)
            plt.plot(x, y_csiborg[i], c="C1", label=label2)
        plt.xlim(0, 75)
        plt.xlabel(r"$r_{1\mathrm{NN}}~[\mathrm{Mpc}]$")
        if kind == "pdf":
            plt.ylabel(r"$p(r_{1\mathrm{NN}})$")
        else:
            plt.ylabel(r"$\mathrm{CDF}(r_{1\mathrm{NN}})$")
            plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.fout, f"1nn_{kind}_{run}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


def plot_significance_hist(simname, run, nsim, nobs, kind, kwargs):
    """Plot a histogram of the significance of the 1NN distance."""
    assert kind in ["kl", "ks"]
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    if kind == "kl":
        x = make_kl(simname, run, nsim, nobs, kwargs)
    else:
        x = make_ks(simname, run, nsim, nobs, kwargs)
        x = numpy.log10(x)
    x = x[numpy.isfinite(x)]

    with plt.style.context(utils.mplstyle):
        plt.figure()
        plt.hist(x, bins="auto")

        if kind == "ks":
            plt.xlabel(r"$\log p$-value of $r_{1\mathrm{NN}}$ distribution")
        else:
            plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"Counts")
        plt.tight_layout()

        for ext in ["png"]:
            if simname == "quijote":
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            fout = join(utils.fout, f"significance_{kind}_{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


def plot_significance_mass(simname, run, nsim, nobs, kind, kwargs):
    """
    Plot significance of the 1NN distance as a function of the total mass.
    """
    assert kind in ["kl", "ks"]
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    x = reader.read_single(simname, run, nsim, nobs)["mass"]
    if kind == "kl":
        y = make_kl(simname, run, nsim, nobs, kwargs)
    else:
        y = make_ks(simname, run, nsim, nobs, kwargs)

    with plt.style.context(utils.mplstyle):
        plt.figure()
        plt.scatter(x, y)

        plt.xscale("log")
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
        if kind == "ks":
            plt.ylabel(r"$p$-value of $r_{1\mathrm{NN}}$ distribution")
            plt.yscale("log")
        else:
            plt.ylabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")

        plt.tight_layout()
        for ext in ["png"]:
            if simname == "quijote":
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            fout = join(utils.fout, f"significance_vs_mass_{kind}_{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


def plot_kl_vs_ks(simname, run, nsim, nobs, kwargs):
    """
    Plot Kullback-Leibler divergence vs Kolmogorov-Smirnov statistic p-value.
    """
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    x = reader.read_single(simname, run, nsim, nobs)["mass"]
    y_kl = make_kl(simname, run, nsim, nobs, kwargs)
    y_ks = make_ks(simname, run, nsim, nobs, kwargs)

    with plt.style.context(utils.mplstyle):
        plt.figure()
        plt.scatter(y_kl, y_ks, c=numpy.log10(x))
        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")

        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"$p$-value of $r_{1\mathrm{NN}}$ distribution")
        plt.yscale("log")

        plt.tight_layout()
        for ext in ["png"]:
            if simname == "quijote":
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            fout = join(utils.fout, f"kl_vs_ks{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    kwargs = {"rmax_radial": 155 / 0.705,
              "nbins_radial": 20,
              "rmax_neighbour": 100.,
              "nbins_neighbour": 150,
              "paths_kind": csiborgtools.paths_glamdring}

    cached_funcs = ["read_dist", "make_kl", "make_ks"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function `{func}`.")
            delete_disk_caches_for_function(func)

    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    run = "mass003"

    # for kind in ["pdf", "cdf"]:
    #     plot_dist(run, kind, kwargs)
    # for kind in ["kl", "ks"]:
    #     # plot_significance_hist("csiborg", run, 7444, nobs=None, kind=kind,
    #     #                        kwargs=kwargs)
    #     plot_significance_mass("quijote", run, 0, nobs=0, kind=kind,
    #                            kwargs=kwargs)

    # plot_significance_mass("quijote", run, 0, nobs=0, kind="ks",
    #                        kwargs=kwargs)

    plot_kl_vs_ks("quijote", run, 0, nobs=0, kwargs=kwargs)
    plot_kl_vs_ks("csiborg", run, 7444, nobs=None, kwargs=kwargs)
