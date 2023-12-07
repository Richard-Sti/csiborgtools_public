# Copyright (C) 2023 Mladen Ivkovic, Richard Stiskalek
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

import copy
import os
from os.path import exists, join
from os import makedirs
from sys import argv
from datetime import datetime

import numpy as np
from joblib import dump, load
from tqdm import trange

errmsg = """

------------------------------------
    mergertree-extract.py
------------------------------------


---------------
    Usage
---------------

This script extracts the masses of clumps and haloes written by the mergertree
patch.
It needs output_XXXXX/mergertree_XXXXX.txtYYYYY and
output_XXXXX/clump_XXXXX.txtYYYYY files to work.
You need to run it from the directory where the output_XXXXX directories are
in.


There are three working modes defined:

1) do for one clump only.
    You need to provide the clump ID you want it done for.
    You can provide a starting directory, but by default the script will
    search for the directory where z = 0.

    run with `python3 mergertree-extract.py <clumpid> [--options] `

    this creates the file mergertree_XXXXX_halo-<halo-ID>.txt. Its contents are
    discussed below.


2) do for one halo.
    You need to provide the halo ID you want it done for, and the flag
    -c or --children.
    The script will by itself find all the child clumps and walk through
    their main branches as well, and write them down.

    run with `python3 mergertree-extract.py <haloid> -c [--options]`
          or `python3 mergertree-extract.py <haloid> --children [--options]`

    this creates the hollowing files:

        - halo_hierarchy_XXXXX-<halo-ID>.txt
            contains the halo ID, how many children it has, and the children
            IDs

        - mergertree_XXXXX_halo-<halo-ID>.txt
            mergertree data for halo that you chose.

        - mergertree_XXXXX_subhalo-<child-ID>.txt
            mergertree data for subhalos of the halo you chose.  One file will
            be created for each subhalo.

        The contents of the mergertree_XXXXX* files are discussed below.


3) do for all haloes
    The script will just walk off all haloes in the z = 0 directory. Note:
    Haloes, not clumps!
    run with `python3 mergertree-extract.py -a [--options]`
          or `python3 mergertree-extract.py --all [--options]`

    This will create the same type of files as in mode (2), just for all
    haloes.


If only an integer is given as cmdline arg, mode (1) [one clump only] will be
run. If no cmd line argument is given, mode (3) [--all] will be run.



---------------
    Output
---------------

the mergertree_XXXXX* files have 6 columns:

snapshot            The snapshot from which this data is taken from

redshift            The redshift of that snapshot

clump_ID            The clump ID of the clump at that snapshot

mass                The mass of the clump at that snapshot, based on what's in
                    the output_XXXXX/mergertree_XXXXX.txtYYYYY files, not the
                    output_XXXXX/clump_XXXXX.txtYYYYY files.

mass_from_mergers   how much mass has been merged into this clump in this
                    snapshot, i.e. the sum of all the clump masses that have
                    been found to merge with this clump at this snapshot. This
                    does not include the mass of clumps which only seem to
                    merge with this clump, but re-emerge later.

mass_from_jumpers   The mass of all clumps that seem to merge with this clump,
                    but re-emerge at a later time.


----------------
    Options
----------------

List of all flags:

Running modes

    -a, --all:      make trees for all clumps in output where z = 0
    -c --children:  make trees for a halo and all its subhaloes. You need to
                    specify which halo via its halo ID.
    -h, --help:     print this help and exit.

Options:
    --start-at=INT      don't start at z = 0 snapshot, but with the specified
                        directory output_00INT.
    --prefix=some/path/ path where you want your output written to.
    -v, --verbose:      be more verbose about what you're doing




-----------------
  Requirements
-----------------

It needs output_XXXXX/mergertree_XXXXX.txtYYYYY and
output_XXXXX/clump_XXXXX.txtYYYYY files to work, which are created using the
mergertree patch in ramses.

Also needs numpy.
"""

###############################################################################
#                             Clump data                                      #
###############################################################################


class ClumpData:
    """
    Data from clump_XXXXX.txt

    Parameters
    ----------
    par : params object
    """
    def __init__(self, par):
        self.clumpids = np.zeros(1)     # clump ID
        self.parent = np.zeros(1)       # parent ID
        self.level = np.zeros(1)        # clump level

    def read_clumpdata(self, par):
        """Reads in the clump data for the z = 0 directory."""
        if par.verbose:
            print("Reading clump data.")

        out = p.z0

        raw_data = [None for i in range(par.ncpu)]
        dirnrstr = str(par.outputnrs[out]).zfill(5)
        dirname = 'output_' + dirnrstr

        i = 0
        for cpu in range(1):
            fname = join(par.workdir, dirname, 'clump_' + dirnrstr + '.dat')
            new_data = np.loadtxt(fname, dtype='int', skiprows=1,
                                  usecols=[0, 1, 2])
            if new_data.ndim == 2:
                raw_data[i] = new_data
                i += 1
            elif new_data.shape[0] == 3:  # if only 1 row is present in file
                raw_data[i] = np.atleast_2d(new_data)
                i += 1

        fulldata = np.concatenate(raw_data[:i], axis=0)
        self.clumpids = fulldata[:, 0]
        self.level = fulldata[:, 1]
        self.parent = fulldata[:, 2]

    def cleanup_clumpdata(self, par, mtd):
        """
        The particle unbinding can remove entire clumps from the catalogue.
        If the option isn't set in the namelist, the clumpfinder output will
        still be made not based on the clumpfinder. If that is the case, the
        clumpfinder catalogue will contain clumps which the mergertree data
        doesn't have, leading to problems. So remove those here.
        """
        for i, c in enumerate(self.clumpids):
            if c not in mtd.descendants[par.z0]:
                self.clumpids[i] = 0
                self.level[i] = 0
                self.parent[i] = -1  # don't make it the same as clumpid

    def find_children(self, clumpid):
        """Find the children for given clump ID."""
        children = []
        last_added = [clumpid]

        loopcounter = 0
        while True:
            loopcounter += 1
            this_level_parents = copy.copy(last_added)
            children += this_level_parents
            last_added = []
            for i, cid in enumerate(self.clumpids):
                if self.parent[i] in this_level_parents and cid != clumpid:
                    last_added.append(cid)

            if len(last_added) == 0:
                break

            if loopcounter == 100:
                print("Finished 100 iterations, we shouldn't be this deep")
                break

        return children[1:]  # don't return top level parent

    def write_children(self, par, clumpid, children):
        """Write the children to file."""
        hfile = join(par.outdir, f"{par.halofilename}-{str(clumpid)}.txt")

        with open(hfile, 'w') as f:
            f.write("# {0:>18} {1:>18} {2:>18}\n".format("halo", "nr_of_children", "children"))  # noqa
            nc = len(children)
            dumpstring = "  {0:18d} {1:18d}".format(clumpid, nc)
            dumpstring = "".join([dumpstring] + [" {0:18d}".format(c) for c in children] + ['\n'])  # noqa
            f.write(dumpstring)


###############################################################################
#                            Constants object                                 #
###############################################################################


class Constants:
    """
    Class holding constants.
    """
    def __init__(self):
        self.Mpc = 3.086e24                 # cm
        self.M_Sol = 1.98855e33             # g
        self.Gyr = (24 * 3600 * 365 * 1e9)  # s
        self.G = 4.492e-15                  # Mpc^3/(M_sol Gyr^2)

        self.H0 = 100                      # km/s/Mpc
        self.omega_m = 0.307000011205673
        self.omega_l = 0.693000018596649
        self.omega_k = 0.0
        self.omega_b = 0.0


###############################################################################
#                             Params object                                   #
###############################################################################


class Params:
    """
    Global parameters to be stored
    """
    def __init__(self):
        # self.workdir = f"/mnt/extraspace/hdesmond/ramses_out_{self.nsim}"
        # self.outdir = f"/mnt/extraspace/rstiskalek/CSiBORG/cleaned_mtree/ramses_out_{self.nsim}"  # noqa
        # if not exists(self.outdir):
        #     makedirs(self.outdir)
        self.lastdir = ""               # last output_XXXXX directory
        self.lastdirnr = -1             # XXXX from lastdir
        self.ncpu = 1                   # Number of CPUs used
        self.noutput = 1                # how many output_XXXXX dirs exist
        self.nout = 1                   # how many outputs we're gonna deal with. (Some might not have merger tree data)  # noqa
        self.outputnrs = None           # numpy array of output numbers
        self.output_lowest = 0          # lowest snapshot number that we're dealing with (>= 1)  # noqa
        self.z0 = 0                     # index of z=0 snapshot (or whichever you want to start with)  # noqa

        # NOTE: params.nout will be defined such that you can easily loop

        self.verbose = False            # verbosity
        self.start_at = 0               # output dir to start with, if given

        self.output_prefix = ""         # user given prefix for output files
        self.outputfilename = ""        # output filename. Stores prefix/mergertree_XXXXX part of name only  # noqa
        self.halofilename = ""          # output filename for halo hierarchy. Stores prefix/halo_hierarchy_XXXXX part of filename only  # noqa

        self.one_halo_only = False      # do the tree for one clump only
        self.halo_and_children = False  # do the tree for one halo, including subhaloes  # noqa
        self.do_all = False             # do for all clumps at z=0 output

        self.clumpid = 0                # which clump ID to work for.
        self.nsim = None

        # Dictionnary of accepted keyword command line arguments
        self.accepted_flags = {
            '-a': self.set_do_all,
            '--all': self.set_do_all,
            '-r': self.set_halo_and_children,
            '--recursive': self.set_halo_and_children,
            '-c': self.set_halo_and_children,
            '--children': self.set_halo_and_children,
            '-h': self.get_help,
            '--help': self.get_help,
            '-v': self.set_verbose,
            '--verbose': self.set_verbose,
            }

        self.accepted_flags_with_args = {
            "--nsim": self.set_nsim,
            '--start-at': self.set_startnr,
            '--prefix': self.set_prefix,
            }

    # -----------------------------
    # Setter methods
    # -----------------------------

    def set_do_all(self):
        self.do_all = True
        return

    def set_halo_and_children(self):
        self.halo_and_children = True
        return

    def get_help(self):
        print(errmsg)
        quit()
        return

    def set_verbose(self):
        self.verbose = True
        return

    def set_startnr(self, arg):
        flag, startnr = arg.split("=")
        try:
            self.start_at = int(startnr)
        except ValueError:
            print("given value for --start-at=INT isn't an integer?")

    def set_prefix(self, arg):
        flag, prefix = arg.split("=")
        #  try:
        self.output_prefix = prefix
        try:
            os.makedirs(self.output_prefix)
        except FileExistsError:
            pass
        return

    def set_nsim(self, arg):
        flag, nsim = arg.split("=")
        try:
            self.nsim = int(nsim)
        except ValueError:
            print("given value for --nsim=INT isn't an integer?")

    def read_cmdlineargs(self):
        """
        Reads in the command line arguments and store them in the
        global_params object.
        """
        nargs = len(argv)
        i = 1  # first cmdlinearg is filename of this file, so skip it

        while i < nargs:
            arg = argv[i]
            arg = arg.strip()
            if arg in self.accepted_flags.keys():
                self.accepted_flags[arg]()
            else:
                for key in self.accepted_flags_with_args.keys():
                    if arg.startswith(key):
                        self.accepted_flags_with_args[key](arg)
                        break
                else:
                    try:
                        self.clumpid = int(arg)
                    except ValueError:
                        print(f"I didn't recognize the argument '{arg}'. Use "
                              "mergertre-extract.py -h or --help to print "
                              "help message.")
                        quit()

            i += 1

        if self.nsim is None:
            raise ValueError("nsim not set. Use --nsim=INT to set it.")

    @property
    def workdir(self):
        return f"/mnt/extraspace/hdesmond/ramses_out_{self.nsim}"

    @property
    def outdir(self):
        fname = f"/mnt/extraspace/rstiskalek/CSiBORG/cleaned_mtree/ramses_out_{self.nsim}"  # noqa
        if not exists(fname):
            makedirs(fname)
        return fname

    def get_output_info(self):
        """
        Read in the output info based on the files in the current working
        directory. Reads in last directory, ncpu, noutputs. Doesn't read
        infofiles.
        """
        # self.workdir = os.getcwd()
        filelist = os.listdir(self.workdir)

        outputlist = []
        for filename in filelist:
            if filename.startswith('output_'):
                outputlist.append(filename)

        if len(outputlist) < 1:
            print("I didn't find any output_XXXXX directories in current "
                  "working directory. Are you in the correct workdir? "
                  "Use mergertree-extract.py -h or --help to print help "
                  "message.")
            quit()

        outputlist.sort()

        self.lastdir = outputlist[-1]
        self.lastdirnr = int(self.lastdir[-5:])
        self.noutput = len(outputlist)

        if (self.start_at > 0):
            # check that directory exists
            startnrstr = str(self.start_at).zfill(5)
            if 'output_' + startnrstr not in outputlist:
                print("Didn't find specified starting directory "
                      f"output_{startnrstr} use mergertree-extract.py -h or "
                      "--help to print help message.")
                quit()

        # read ncpu from infofile in last output directory
        infofile = join(self.workdir, self.lastdir,
                        f"info_{self.lastdir[-5:]}.txt")
        with open(infofile, 'r') as f:
            ncpuline = f.readline()
            line = ncpuline.split()
            self.ncpu = int(line[-1])

    def setup_and_checks(self, sd):
        """
        Do checks and additional setups once you have all the cmd line args and
        output infos

        Parameters
        ----------
        sd: snapshotdata object
        """
        # set running mode
        if not self.do_all:
            if self.clumpid <= 0:
                print("No or wrong clump id given. Setting the --all mode.")
                self.set_do_all()
            else:
                if not self.halo_and_children:
                    self.one_halo_only = True

        # generate list of outputdirnumbers
        startnr = self.lastdirnr
        self.outputnrs = np.array(range(startnr, startnr - self.noutput, -1))

        # find starting output directory
        self.z0 = np.argmin(np.absolute(sd.redshift))

        if self.start_at > 0:
            # replace z0 dir with starting dir
            self.z0 = self.lastdirnr - self.start_at

        # generate output filename
        dirnrstr = str(self.outputnrs[self.z0]).zfill(5)
        fname = "mergertree_" + dirnrstr
        self.outputfilename = join(self.output_prefix, fname)

        # generate halo output filename
        fname = "halo_hierarchy_" + dirnrstr
        self.halofilename = join(self.output_prefix, fname)

        # rename output_prefix to something if it wasn't set
        if self.output_prefix == "":
            self.output_prefix = os.path.relpath(self.workdir)

        # find self.nout; i.e. how many outputs we are actually going to have
        for out in range(self.noutput - 1, -1, -1):
            dirnrstr = str(self.outputnrs[out]).zfill(5)
            mtreefile = join(self.workdir,
                             f"output_{dirnrstr}",
                             f"mergertree_{dirnrstr}.dat")

            if os.path.exists(mtreefile):
                print("Loading mergertree data from ", mtreefile)
                # if there is a file, this is lowest snapshot number directory
                # that we'll be dealing with, and hence will have the highest
                # index number in the arrays I'm using

                # NOTE: params.nout will be defined such that you can easily
                # loop for out in range(p.z0, p.nout)
                self.nout = out + 1
                break

    def print_params(self):
        """Prints out the parameters that are set."""
        if self.do_all:
            print("Working mode:             all clumps")
        else:
            if self.halo_and_children:
                print("Working mode:             halo", self.clumpid, "and its children")  # noqa
            else:
                print("Working mode:             clump ", self.clumpid)

        print("workdir:                 ", self.workdir)
        print("snapshot of tree root:   ", self.outputnrs[self.z0])
        print("p.one_halo_only          ", p.one_halo_only)
        print("p.do_all                 ", p.do_all)
        print("p.halo_and_children      ", p.halo_and_children)
        print("p.one_halo_only          ", p.one_halo_only)


###############################################################################
#                             Merger tree data                                #
###############################################################################


class MTreeData:
    """
    Merger tree data lists

    Parameters
    ----------
    par : params object
    """
    def __init__(self, par):
        self.progenitors = [np.zeros(1) for i in range(par.noutput)]            # progenitor IDs  # noqa
        self.descendants = [np.zeros(1) for i in range(par.noutput)]            # descendant IDs  # noqa
        self.progenitor_outputnrs = [np.zeros(1) for i in range(par.noutput)]   # snapshot number of progenitor  # noqa
        self.mass = [np.zeros(1) for i in range(par.noutput)]                   # descendant mass  # noqa
        self.mass_to_remove = [np.zeros(1) for i in range(par.noutput)]         # descendant mass  # noqa

    def read_mergertree_data(self, par, sd):
        """Reads in mergertree data."""

        if par.verbose:
            print("Reading in mergertree data")

        # Preparation

        # define new datatype for mergertree output
        mtree = np.dtype([('clump', 'i4'),
                          ('prog', 'i4'),
                          ('prog_outnr', 'i4'),
                          ('mass', 'f8'),
                          ('npart', 'f8'),
                          ('x', 'f8'),
                          ('y', 'f8'),
                          ('z', 'f8'),
                          ('vx', 'f8'),
                          ('vy', 'f8'),
                          ('vz', 'f8')
                          ])

        # ---------------------------
        # Loop over directories
        # ---------------------------

        startnr = par.lastdirnr
        # READ THE ONES BEFORE z0 TOO!
        for output in trange(par.nout, desc="Reading merger"):
            dirnr = str(startnr - output).zfill(5)
            srcdir = 'output_' + dirnr

            fnames = [srcdir + '/' + "mergertree_" + dirnr + '.dat']
            fnames[0] = join(par.workdir, fnames[0])

            datalist = [np.zeros((1, 3)) for i in range(par.ncpu)]
            i = 0
            nofile = 0
            for f in fnames:
                if os.path.exists(f):
                    datalist[i] = np.atleast_1d(np.genfromtxt(f, dtype=mtree,
                                                              skip_header=1))
                    i += 1
                else:
                    nofile += 1

            if nofile == p.ncpu:
                print("Didn't find any mergertree data in", srcdir)

            # ---------------------------------
            # Sort out data
            # ---------------------------------
            if i > 0:
                fulldata = np.concatenate(datalist[:i], axis=0)

                self.descendants[output] = fulldata[:]['clump']
                self.progenitors[output] = fulldata[:]['prog']
                self.progenitor_outputnrs[output] = fulldata[:]['prog_outnr']
                self.mass[output] = fulldata[:]['mass']
                #  self.npart[output] = fulldata[:]['npart']
                #  self.x[output] = fulldata[:]['x']
                #  self.y[output] = fulldata[:]['y']
                #  self.z[output] = fulldata[:]['z']
                #  self.vx[output] = fulldata[:]['vx']
                #  self.vy[output] = fulldata[:]['vy']
                #  self.vz[output] = fulldata[:]['vz']

        # --------------------------------------
        # Transform units to physical units
        # --------------------------------------

        # transform units to physical units
        for i in range(len(self.descendants)):
            self.mass[i] *= sd.unit_m[i]
            #  self.x[i] *= sd.unit_l[i] # only transform later when needed; Need to check for periodicity first!  # noqa
            #  self.y[i] *= sd.unit_l[i]
            #  self.z[i] *= sd.unit_l[i]
            #  self.vx[i] *= sd.unit_l[i]/sd.unit_t[i]
            #  self.vy[i] *= sd.unit_l[i]/sd.unit_t[i]
            #  self.vz[i] *= sd.unit_l[i]/sd.unit_t[i]

    def clean_up_jumpers(self, par):
        """
        Remove jumpers from the merger list. Take note of how much mass should
        be removed from the descendant because the jumper is to be removed.
        """
        # First initialize mass_to_remove arrays
        self.mass_to_remove = [np.zeros(self.descendants[out].shape)
                               for out in range(par.noutput)]
        nreplaced = 0
        for out in trange(par.nout + par.z0 - 1, desc="Cleaning jumpers"):
            for i, pr in enumerate(self.progenitors[out]):
                if pr < 0:
                    # Subtract 1 here from snapind:
                    # progenitor_outputnrs gives the snapshot number where the
                    # jumper was a descendant for the last time
                    # so you need to overwrite the merging one snapshot later,
                    # where the clump is the progenitor
                    snapind = get_snap_ind(p, self.progenitor_outputnrs[out][i]) - 1  # noqa

                    # NOTE bottleneck
                    jumpind = self.progenitors[snapind] == -pr

                    # NOTE bottleneck
                    # find index of descendant into which this clump will
                    # appearingly merge into
                    mergerind = self.descendants[snapind] == - self.descendants[snapind][jumpind]  # noqa
                    # overwrite merging event so it won't count
                    self.descendants[snapind][jumpind] = 0

                    # find mass of jumper in previous snapshot
                    jumpmassind = self.descendants[snapind + 1] == -pr
                    # note how much mass might need to be removed for whatever
                    # you need it
                    self.mass_to_remove[snapind][mergerind] += self.mass[snapind + 1][jumpmassind]  # noqa

                    nreplaced += 1

        print("Cleaned out", nreplaced, "jumpers")

    def get_tree(self, par, tree, sd, clumpid):
        """Follow the main branch down."""
        if par.verbose:
            print("Computing tree for clump", clumpid)

        dind = self.descendants[par.z0] == clumpid
        desc_snap_ind = p.z0
        desc = self.descendants[p.z0][dind]
        prog = self.progenitors[p.z0][dind]

        def get_prog_indices(prog, desc_snap_ind):
            """
            Compute snapshot index at which given progenitor has been a
            descendant and its index in the array

            prog:           progenitor ID
            desc_snap_ind:  snapshot index of descendant of given prog

            returns:
            p_snap_ind:     snapshot index of the progenitor
            pind:           progenitor index (np.array mask) of progenitor in
                            array where it is descendant
            """
            if prog > 0:  # if progenitor isn't jumper
                # find progenitor's index in previous snapshot
                p_snap_ind = desc_snap_ind + 1
                pind = self.descendants[p_snap_ind] == prog

            elif prog < 0:
                p_snap_ind = get_snap_ind(
                    par, self.progenitor_outputnrs[desc_snap_ind][dind])
                pind = self.descendants[p_snap_ind] == -prog

            return p_snap_ind, pind

        while True:
            # first calculate merger mass
            mergers = self.descendants[desc_snap_ind] == -desc
            mergermass = 0.0
            if mergers.any():
                for m in self.progenitors[desc_snap_ind][mergers]:
                    # find mass of merger. That's been written down at the
                    # place where merger was descendant.
                    m_snap_ind, mergerind = get_prog_indices(m, desc_snap_ind)
                    mergermass += self.mass[m_snap_ind][mergerind]

            # add the descendant to the tree
            tree.add_snap(par.outputnrs[desc_snap_ind],
                          sd.redshift[desc_snap_ind], desc,
                          self.mass[desc_snap_ind][dind], mergermass,
                          self.mass_to_remove[desc_snap_ind][dind])

            # now descend down the main branch
            if prog != 0:
                p_snap_ind, pind = get_prog_indices(prog, desc_snap_ind)
            else:
                # stop at progenitor = 0
                break

            # prepare for next round
            desc_snap_ind = p_snap_ind
            dind = pind
            desc = abs(prog)
            prog = self.progenitors[p_snap_ind][pind]


###############################################################################
#                             Snapshot data                                   #
###############################################################################


class SnapshotData():
    """Snapshot specific data"""
    def __init__(self, par):
        # read in
        self.aexp = np.zeros(par.noutput)
        self.unit_l = np.zeros(par.noutput)
        self.unit_m = np.zeros(par.noutput)
        self.unit_t = np.zeros(par.noutput)
        self.unit_dens = np.zeros(par.noutput)
        # to be computed
        self.redshift = np.zeros(par.noutput)  # z

    def read_infofiles(self, par, const):
        """Read the info_XXXXX.txt files."""
        if par.verbose:
            print("Reading info files.")

        startnr = par.lastdirnr

        for output in range(p.noutput):
            # Start with last directory (e.g. output_00060),
            # work your way to first directory (e.g. output_00001)
            # p.z0 isn't decided yet, so just read in everything here.
            dirnr = str(startnr - output).zfill(5)
            srcdir = 'output_' + dirnr

            try:
                # ------------------------------------------------------
                # get time, redshift, and units even for output_00001
                # ------------------------------------------------------
                fileloc = srcdir + '/info_' + dirnr + '.txt'
                fileloc = join(par.workdir, fileloc)
                infofile = open(fileloc)
                for i in range(9):
                    infofile.readline()  # skip first 9 lines

                # get expansion factor
                aline = infofile.readline()
                astring, equal, aval = aline.partition("=")
                afloat = float(aval)
                sd.aexp[output] = afloat

                for i in range(5):
                    infofile.readline()  # skip 5 lines

                # get unit_l
                unitline = infofile.readline()
                unitstring, equal, unitval = unitline.partition("=")
                unitfloat = float(unitval)
                sd.unit_l[output] = unitfloat

                # get unit_dens
                unitline = infofile.readline()
                unitstring, equal, unitval = unitline.partition("=")
                unitfloat = float(unitval)
                sd.unit_dens[output] = unitfloat

                # get unit_t
                unitline = infofile.readline()
                unitstring, equal, unitval = unitline.partition("=")
                unitfloat = float(unitval)
                sd.unit_t[output] = unitfloat

                infofile.close()

            except IOError:  # If file doesn't exist
                print("Didn't find any info data in ", srcdir)
                break

        self.unit_m = self.unit_dens * self.unit_l ** 3 / const.M_Sol
        self.unit_l /= const.Mpc
        self.unit_t /= const.Gyr

        self.redshift = 1. / self.aexp - 1

###############################################################################
#                             Tree object                                     #
###############################################################################


class Tree:
    """
    Holds tree result data. It's not really a tree, it's just the values along
    the main branch, but let's call it a tree anyway. Sue me.

    Parameters
    ----------
    nelements : int
        Estimate for how many snapshots you need to allocate space for.
    """
    def __init__(self, nelements):
        self.n = 0                                              # number of elements in tree  # noqa
        self.snapshotnr = -np.ones(nelements, dtype=int)        # snapshot number of array values  # noqa
        self.redshift = -np.ones(nelements, dtype=float)        # redshift at that snapshot  # noqa
        self.clumpids = -np.ones(nelements, dtype=int)          # clump id of halo in that snapshot  # noqa
        self.mass = np.zeros(nelements, dtype=float)            # mass at that snapshot  # noqa
        self.mergermass = np.zeros(nelements, dtype=float)      # sum of mass of swallowed up clumps  # noqa
        self.mass_to_remove = np.zeros(nelements, dtype=float)  # sum of mass of swallowed up clumps  # noqa

    def add_snap(self, nr, z, ID, m, mm, mdel):
        """Add new result."""
        n = self.n
        self.snapshotnr[n] = nr
        self.redshift[n] = z
        self.clumpids[n] = ID
        self.mass[n] = m
        self.mergermass[n] = mm
        self.mass_to_remove[n] = mdel
        self.n += 1

    def write_tree(self, par, case='halo'):
        """Write the results to file."""
        resfile = join(
            par.outdir,
            f"{par.outputfilename}_{case}-{str(self.clumpids[0])}.txt")

        with open(resfile, 'w') as f:
            f.write('# {0:>12} {1:>12} {2:>16} {3:>18} {4:>18} {5:>18}\n'.format(  # noqa
                "snapshot", "redshift", "clump_ID", "mass[M_sol]",
                "mass_from_mergers", "mass_from_jumpers"))

            for i in range(self.n):
                f.write('  {0:12d} {1:12.4f} {2:16d} {3:18.6e} {4:18.6e} {5:18.6e}\n'.format(  # noqa
                    self.snapshotnr[i], self.redshift[i], self.clumpids[i],
                    self.mass[i], self.mergermass[i], self.mass_to_remove[i]))

        return


def get_snap_ind(p, snap):
    """
    Computes the snapshot index in mtreedata/halodata/snapshotdata arrays for a
    given snapshot number snap
    """
    return (p.noutput - snap).item()


if __name__ == '__main__':

    p = Params()
    c = Constants()

    # Read cmdlineargs, available output, get global parameters
    p.read_cmdlineargs()
    p.get_output_info()

    sd = SnapshotData(p)
    sd.read_infofiles(p, c)

    # finish setup
    p.setup_and_checks(sd)
    p.print_params()

    # now read in mergertree data
    fname = join(p.outdir, "mtreedata.p")
    if exists(fname):
        print(f"{datetime.now()}: loading mergertree data from `{fname}`.",
              flush=True)
        mtd = load(fname)
        print(f"{datetime.now()}: finished loading mergertree data from `{fname}`.",  # noqa
              flush=True)
    else:
        print("Generating mergertree data.", flush=True)
        mtd = MTreeData(p)
        mtd.read_mergertree_data(p, sd)
        # clean up jumpers
        mtd.clean_up_jumpers(p)

        print("Saving mergertree data.", flush=True)
        dump(mtd, fname)

    # read in clump data if required
    if p.do_all or p.halo_and_children:
        cd = ClumpData(p)
        cd.read_clumpdata(p)

        # clean up halo catalogue
        cd.cleanup_clumpdata(p, mtd)

        # find children, and write them down
        if p.verbose:
            print("Searching for child clumps.")

        if p.halo_and_children:
            children = cd.find_children(p.clumpid)
            cd.write_children(p, p.clumpid, children)

        if p.do_all:
            is_halo = cd.clumpids == cd.parent
            childlist = [None for c in cd.clumpids[is_halo]]
            for i, halo in enumerate(cd.clumpids[is_halo]):
                children = cd.find_children(halo)
                cd.write_children(p, halo, children)
                childlist[i] = children

    # finally, get the bloody tree

    if p.one_halo_only:
        newtree = Tree(p.nout)
        mtd.get_tree(p, newtree, sd, p.clumpid)
        newtree.write_tree(p, 'halo')

    if p.halo_and_children:
        newtree = Tree(p.nout)
        mtd.get_tree(p, newtree, sd, p.clumpid)
        newtree.write_tree(p, 'halo')

        for c in children:
            newtree = Tree(p.nout)
            mtd.get_tree(p, newtree, sd, c)
            newtree.write_tree(p, 'subhalo')

    if p.do_all:
        for i, halo in enumerate(cd.clumpids[is_halo]):
            newtree = Tree(p.nout)
            mtd.get_tree(p, newtree, sd, halo)
            newtree.write_tree(p, 'halo')

            for c in childlist[i]:
                newtree = Tree(p.nout)
                mtd.get_tree(p, newtree, sd, c)
                newtree.write_tree(p, 'subhalo')

    print('Finished.')
