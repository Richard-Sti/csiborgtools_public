# Copyright (C) 2022 Richard Stiskalek
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
"""A script to calculate the KNN-CDF for a set of CSiBORG halo catalogues."""
from os.path import join
from warnings import warn
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from mpi4py import MPI
from TaskmasterMPI import master_process, worker_process
import numpy
from sklearn.neighbors import NearestNeighbors
import joblib
import yaml
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


###############################################################################
#                            MPI and arguments                                #
###############################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

parser = ArgumentParser()
parser.add_argument("--runs", type=str, nargs="+")
args = parser.parse_args()
with open('../scripts/knn_auto.yml', 'r') as file:
    config = yaml.safe_load(file)

Rmax = 155 / 0.705  # Mpc (h = 0.705) high resolution region radius
totvol = 4 * numpy.pi * Rmax**3 / 3
minmass = 1e12
ics = [7444, 7468, 7492, 7516, 7540, 7564, 7588, 7612, 7636, 7660, 7684,
       7708, 7732, 7756, 7780, 7804, 7828, 7852, 7876, 7900, 7924, 7948,
       7972, 7996, 8020, 8044, 8068, 8092, 8116, 8140, 8164, 8188, 8212,
       8236, 8260, 8284, 8308, 8332, 8356, 8380, 8404, 8428, 8452, 8476,
       8500, 8524, 8548, 8572, 8596, 8620, 8644, 8668, 8692, 8716, 8740,
       8764, 8788, 8812, 8836, 8860, 8884, 8908, 8932, 8956, 8980, 9004,
       9028, 9052, 9076, 9100, 9124, 9148, 9172, 9196, 9220, 9244, 9268,
       9292, 9316, 9340, 9364, 9388, 9412, 9436, 9460, 9484, 9508, 9532,
       9556, 9580, 9604, 9628, 9652, 9676, 9700, 9724, 9748, 9772, 9796,
       9820, 9844]
dumpdir = "/mnt/extraspace/rstiskalek/csiborg/knn"
fout = join(dumpdir, "auto", "knncdf_{}_{}.p")
paths = csiborgtools.read.CSiBORGPaths()
knncdf = csiborgtools.clustering.kNN_CDF()

###############################################################################
#                                 Analysis                                    #
###############################################################################

def read_single(selection, cat):
    """Positions for single catalogue auto-correlation."""
    mmask = numpy.ones(len(cat), dtype=bool)
    pos = cat.positions(False)
    # Primary selection
    psel = selection["primary"]
    pmin, pmax = psel.get("min", None), psel.get("max", None)
    if pmin is not None:
        mmask &= (cat[psel["name"]] >= pmin)
    if pmax is not None:
        mmask &= (cat[psel["name"]] < pmax)
    pos = pos[mmask, ...]

    # Secondary selection
    if "secondary" not in selection:
        return pos
    smask = numpy.ones(pos.shape[0], dtype=bool)
    ssel = selection["secondary"]
    smin, smax = ssel.get("min", None), ssel.get("max", None)
    prop = cat[ssel["name"]][mmask]
    if ssel.get("toperm", False):
        prop = numpy.random.permutation(prop)
    if ssel.get("marked", True):
        x = cat[psel["name"]][mmask]
        prop = csiborgtools.clustering.normalised_marks(
            x, prop, nbins=config["nbins_marks"])

    if smin is not None:
        smask &= (prop >= smin)
    if smax is not None:
        smask &= (prop < smax)

    return pos[smask, ...]

def do_auto(run, cat, ic):
    """Calculate the kNN-CDF single catalgoue autocorrelation."""
    _config = config.get(run, None)
    if _config is None:
        warn("No configuration for run {}.".format(run))
        return

    rvs_gen = csiborgtools.clustering.RVSinsphere(Rmax)
    pos = read_single(_config, cat)
    knn = NearestNeighbors()
    knn.fit(pos)
    rs, cdf = knncdf(
        knn, rvs_gen=rvs_gen, nneighbours=config["nneighbours"],
        rmin=config["rmin"], rmax=config["rmax"],
        nsamples=int(config["nsamples"]), neval=int(config["neval"]),
        batch_size=int(config["batch_size"]), random_state=config["seed"])

    joblib.dump({"rs": rs, "cdf": cdf, "ndensity": pos.shape[0] / totvol},
                fout.format(str(ic).zfill(5), run))

def do_cross_rand(run, cat, ic):
    """Calculate the kNN-CDF cross catalogue random correlation."""
    _config = config.get(run, None)
    if _config is None:
        warn("No configuration for run {}.".format(run))
        return

    rvs_gen = csiborgtools.clustering.RVSinsphere(Rmax)
    knn1, knn2 = NearestNeighbors(), NearestNeighbors()

    pos1 = read_single(_config, cat)
    knn1.fit(pos1)

    pos2 = rvs_gen(pos1.shape[0])
    knn2.fit(pos2)

    rs, cdf0, cdf1, joint_cdf = knncdf.joint(
        knn1, knn2, rvs_gen=rvs_gen, nneighbours=int(config["nneighbours"]),
        rmin=config["rmin"], rmax=config["rmax"],
        nsamples=int(config["nsamples"]), neval=int(config["neval"]),
        batch_size=int(config["batch_size"]), random_state=config["seed"])
    corr = knncdf.joint_to_corr(cdf0, cdf1, joint_cdf)

    joblib.dump({"rs": rs, "corr": corr}, fout.format(str(ic).zfill(5), run))



def do_runs(ic):
    cat = csiborgtools.read.HaloCatalogue(ic, paths, max_dist=Rmax,
                                          min_mass=minmass)
    for run in args.runs:
        if "random" in run:
            do_cross_rand(run, cat, ic)
        else:
            do_auto(run, cat, ic)


###############################################################################
#                             MPI task delegation                             #
###############################################################################


if nproc > 1:
    if rank == 0:
        tasks = deepcopy(ics)
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_runs, comm, verbose=False)
else:
    tasks = deepcopy(ics)
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task))
        do_runs(task)
comm.Barrier()


if rank == 0:
    print("{}: all finished.".format(datetime.now()))
quit()  # Force quit the script
