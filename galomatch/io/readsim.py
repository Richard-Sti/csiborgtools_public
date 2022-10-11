# PUT HARRY HERE
import numpy
from scipy.io import FortranFile
from os import listdir
from os.path import join
from tqdm import tqdm


F16 = numpy.float16
F32 = numpy.float32
F64 = numpy.float64
INT32 = numpy.int32


def get_sim_path(n, fname="ramses_out_{}", srcdir="/mnt/extraspace/hdesmond"):
    """
    Get a path to a CSiBORG simulation.

    Parameters
    ----------
    n : int
        The index of the initial conditions (IC) realisation.
    fname : str, optional
        The file name. By default `ramses_out_{}`, where `n` is the IC index.
    srcdir : str, optional
        The file path to the folder where realisations of the ICs are stored.

    Returns
    -------
    path : str
        The complete path to the `n`th CSiBORG simulation.
    """
    return join(srcdir, fname.format(n))


def open_particle(n, simpath, verbose=True):
    """
    Open particle files to a given CSiBORG simulation.

    Parameters
    ----------
    n : int
        The index of a redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    nparts : 1-dimensional array
        Number of parts assosiated with each CPU.
    partfiles : list of `scipy.io.FortranFile`
        Opened part files.
    """
    # Zeros filled snapshot number and the snapshot path
    nout = str(n).zfill(5)
    snappath = join(simpath, "output_{}".format(nout))
    infopath = join(snappath, "info_{}.txt".format(nout))

    with open(infopath, "r") as f:
        ncpu = int(f.readline().split()[-1])
    if verbose:
        print("Reading in output `{}` with ncpu = `{}`.".format(nout, ncpu))

    # Check whether the unbinding file exists.
    snapdirlist = listdir(snappath)
    unbinding_file = "unbinding_{}.out00001".format(nout)
    if unbinding_file not in snapdirlist:
        raise FileNotFoundError(
            "Couldn't find `{}` in `{}`. Use mergertreeplot.py -h or --help to "
            "print help message.".format(unbinding_file, snappath))

    # First read the headers. Reallocate arrays and fill them.
    nparts = numpy.zeros(ncpu, dtype=int)
    partfiles = [None] * ncpu
    for cpu in range(ncpu):
        cpu_str = str(cpu + 1).zfill(5)
        fpath = join(snappath, "part_{}.out{}".format(nout, cpu_str))

        f = FortranFile(fpath)
        # Read in this order
        ncpuloc = f.read_ints()
        if ncpuloc != ncpu:
            raise ValueError("`ncpu = {}` of `{}` disagrees with `ncpu = {}` "
                             "of `{}`.".format(ncpu, infopath, ncpuloc, fpath))
        ndim = f.read_ints()
        nparts[cpu] = f.read_ints()
        localseed = f.read_ints()
        nstar_tot = f.read_ints()
        mstar_tot = f.read_reals('d')
        mstar_lost = f.read_reals('d')
        nsink = f.read_ints()

        partfiles[cpu] = f

    return nparts, partfiles


def read_sp(dtype, partfile):
    """
    Utility function to read a single particle file, depending on the dtype.

    Parameters
    ----------
    dtype : str
        The dtype of the part file to be read now.
    partfile : `scipy.io.FortranFile`
        Part file to read from.

    Returns
    -------
    out : 1-dimensional array
        The data read from the part file.
    n : int
        The index of the initial conditions (IC) realisation.
    simpath : str
        The complete path to the CSiBORG simulation.
    """
    if dtype in [F16, F32, F64]:
        return partfile.read_reals('d')
    elif dtype in [INT32]:
        return partfile.read_ints()
    else:
        raise TypeError("Unexpected dtype `{}`.".format(dtype))


def nparts_to_start_ind(nparts):
    """
    Convert `nparts` array to starting indices in a pre-allocated array for looping over the CPU number.

    Parameters
    ----------
    nparts : 1-dimensional array
        Number of parts assosiated with each CPU.

    Returns
    -------
    start_ind : 1-dimensional array
        The starting indices calculated as a cumulative sum starting at 0.
    """
    return numpy.hstack([[0], numpy.cumsum(nparts[:-1])])


def read_particle(pars_extract, n, simpath, verbose=True):
    """
    Read particle files of a simulation at a given snapshot and return
    values of `pars_extract`.

    Parameters
    ----------
    pars_extract : list of str
        Parameters to be extacted.
    n : int
        The index of the redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.
    verbose : bool, optional
        Verbosity flag while for reading the CPU outputs.

    Returns
    -------
    out : structured array
        The data read from the particle file.
    """
    # Open the particle files
    nparts, partfiles = open_particle(n, simpath)
    if verbose:
        print("Opened {} particle files.".format(nparts.size))
    ncpu = nparts.size
    # Order in which the particles are written in the FortranFile
    forder = [("x", F16), ("y", F16), ("z", F16),
              ("vx", F16), ("vy", F16), ("vz", F16),
              ("M", F32), ("ID", INT32), ("level", INT32)]
    fnames = [fp[0] for fp in forder]
    fdtypes = [fp[1] for fp in forder]
    # Check there are no strange parameters
    for p in pars_extract:
        if p not in fnames:
            raise ValueError("Undefined parameter `{}`. Must be one of `{}`."
                             .format(p, fnames))

    npart_tot = numpy.sum(nparts)
    # A dummy array is necessary for reading the fortran files.
    dum = numpy.full(npart_tot, numpy.nan, dtype=F16)
    # These are the data we read along with types
    dtype = {"names": pars_extract,
             "formats": [forder[fnames.index(p)][1] for p in pars_extract]}
    # Allocate the output structured array
    out = numpy.full(npart_tot, numpy.nan, dtype)
    start_ind = nparts_to_start_ind((nparts))
    iters = tqdm(range(ncpu)) if verbose else range(ncpu)
    for cpu in iters:
        i = start_ind[cpu]
        j = nparts[cpu]
        for (fname, fdtype) in zip(fnames, fdtypes):
            if fname in pars_extract:
                out[fname][i:i + j] = read_sp(fdtype, partfiles[cpu])
            else:
                dum[i:i + j] = read_sp(fdtype, partfiles[cpu])

    return out


def read_unbibding():
    pass
