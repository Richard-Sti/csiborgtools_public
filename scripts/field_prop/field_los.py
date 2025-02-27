# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
MPI script to interpolate the density and velocity fields along the line of
sight.
"""
import sys
from argparse import ArgumentParser
from datetime import datetime
from gc import collect
from os import makedirs, remove, rmdir
from os.path import exists, join
from warnings import warn

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from h5py import File
from mpi4py import MPI
from numba import jit
from scipy.interpolate import interp1d
from taskmaster import work_delegation  # noqa

import csiborgtools

sys.path.append("../")
from utils import get_nsims  # noqa

###############################################################################
#                             I/O functions                                   #
###############################################################################


def make_spacing(rmax, dr, dense_mu_min, dense_mu_max, dmu, Om0):
    """
    Make radial spacing that at low distance is with constant spacing in
    distance modulus and at higher distances is with constant spacing in
    comoving distance.
    """
    # Create interpolant to go from distance modulus to comoving distance.
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)

    z_range = np.linspace(0, 0.1, 1000000)[1:]
    r_range = cosmo.comoving_distance(z_range).value
    mu_range = cosmo.distmod(z_range).value

    mu2r = interp1d(mu_range, r_range, kind='cubic')

    # Create the spacing in distance modulus.
    mu = np.arange(dense_mu_min, dense_mu_max, dmu)
    rmin_dense = mu2r(np.min(mu))
    rmax_dense = mu2r(np.max(mu))

    # Create the spacing in comoving distance below and above.
    rlow = np.arange(0, rmin_dense, dr)
    rmed = mu2r(mu)
    rhigh = np.arange(rmax_dense, rmax, dr)[1:]

    # Combine the spacings.
    return np.hstack([rlow, rmed, rhigh])


def get_los(catalogue_name, simname, comm):
    """
    Get the line of sight RA/dec coordinates for the given catalogue.

    Parameters
    ----------
    catalogue_name : str
        Catalogue name.
    simname : str
        Simulation name.
    comm : mpi4py.MPI.Comm
        MPI communicator.

    Returns
    -------
    pos : 2-dimensional array
        RA/dec coordinates of the line of sight.
    """
    if comm.Get_rank() == 0:
        folder = "/mnt/extraspace/rstiskalek/catalogs"

        if catalogue_name in ["LOSS", "Foundation", "SFI_gals",
                              "SFI_gals_masked", "SFI_groups", "2MTF",
                              "Pantheon+"]:
            fpath = join(folder, "PV_compilation.hdf5")
            with File(fpath, 'r') as f:
                grp = f[catalogue_name]
                RA = grp["RA"][:]
                dec = grp["DEC"][:]
        elif catalogue_name == "A2":
            fpath = join(folder, "A2.h5")
            with File(fpath, 'r') as f:
                RA = f["RA"][:]
                dec = f["DEC"][:]
        elif "CB2_" in catalogue_name:
            kind = catalogue_name.split("_")[-1]
            fname = f"/mnt/extraspace/rstiskalek/catalogs/PV_mock_CB2_17417_{kind}.hdf5"  # noqa
            with File(fname, 'r') as f:
                RA = f["RA"][:]
                dec = f["DEC"][:]
        elif catalogue_name == "UPGLADE":
            fname = "/mnt/users/rstiskalek/csiborgtools/data/upglade_all_z0p05_new_PROCESSED.h5"  # noqa
            with File(fname, 'r') as f:
                RA = f["RA"][:]
                dec = f["DEC"][:]
        elif catalogue_name == "CF4_GroupAll":
            fname = "/mnt/extraspace/rstiskalek/catalogs/PV/CF4/CF4_GroupAll.hdf5"  # noqa
            with File(fname, 'r') as f:
                RA = f["RA"][:]
                dec = f["DE"][:]
        elif catalogue_name == "CF4_TFR":
            fname = "/mnt/extraspace/rstiskalek/catalogs/PV/CF4/CF4_TF-distances.hdf5"  # noqa
            with File(fname, 'r') as f:
                RA = f["RA"][:] * 360 / 24  # Convert to degrees from hours.
                dec = f["DE"][:]
        else:
            raise ValueError(f"Unknown catalogue name: `{catalogue_name}`.")

        if comm.Get_rank() == 0:
            print(f"The dataset contains {len(RA)} objects.")

        if simname in ["Carrick2015", "Lilow2024"]:
            # The Carrick+2015 and Lilow+2024 are in galactic coordinates, so
            # we need to convert the RA/dec to galactic coordinates.
            c = SkyCoord(ra=RA*u.degree, dec=dec*u.degree, frame='icrs')
            pos = np.vstack((c.galactic.l, c.galactic.b)).T
        elif simname in ["CF4", "CLONES"]:
            # CF4 and CLONES fields are in supergalactic coordinates.
            c = SkyCoord(ra=RA*u.degree, dec=dec*u.degree, frame='icrs')
            pos = np.vstack((c.supergalactic.sgl, c.supergalactic.sgb)).T
        else:
            pos = np.vstack((RA, dec)).T
    else:
        pos = None

    return comm.bcast(pos, root=0)


def get_field(simname, nsim, kind, MAS, grid):
    """
    Get the field from the simulation.

    Parameters
    ----------
    simname : str
        Simulation name.
    nsim : int
        IC realisation index.
    kind : str
        Field kind. Either `density` or `velocity`.
    MAS : str
        Mass assignment scheme.
    grid : int
        Grid resolution.

    Returns
    -------
    field : n-dimensional array
    """
    # Open the field reader.
    if simname == "csiborg1":
        field_reader = csiborgtools.read.CSiBORG1Field(nsim)
    elif "csiborg2_" in simname:
        simkind = simname.split("_")[-1]
        field_reader = csiborgtools.read.CSiBORG2Field(nsim, simkind)
    elif simname == "csiborg2X":
        field_reader = csiborgtools.read.CSiBORG2XField(nsim, version=0)
    elif simname == "manticore_2MPP_N128_DES_V1":
        field_reader = csiborgtools.read.CSiBORG2XField(nsim, version=1)
    elif simname == "CLONES":
        field_reader = csiborgtools.read.CLONESField(nsim)
    elif simname == "Carrick2015":
        folder = "/mnt/extraspace/rstiskalek/catalogs"
        warn(f"Using local paths from `{folder}`.", RuntimeWarning)
        if kind == "density":
            fpath = join(folder, "twompp_density_carrick2015.npy")
            return np.load(fpath).astype(np.float32)
        elif kind == "velocity":
            fpath = join(folder, "twompp_velocity_carrick2015.npy")
            field = np.load(fpath).astype(np.float32)

            # Because the Carrick+2015 data is in the following form:
            # "The velocities are predicted peculiar velocities in the CMB
            # frame in Galactic Cartesian coordinates, generated from the
            # \(\delta_g^*\) field with \(\beta^* = 0.43\) and an external
            # dipole \(V_\mathrm{ext} = [89,-131,17]\) (Carrick et al Table 3)
            # has already been added.""
            field[0] -= 89
            field[1] -= -131
            field[2] -= 17
            field /= 0.43

            return field
        else:
            raise ValueError(f"Unknown field kind: `{kind}`.")
    elif simname == "CF4":
        folder = "/mnt/extraspace/rstiskalek/catalogs/CF4"
        warn(f"Using local paths from `{folder}`.", RuntimeWarning)

        if kind == "density":
            fpath = join(folder, f"CF4_new_128-z008_realization{nsim}_delta.fits")     # noqa
        elif kind == "velocity":
            fpath = join(folder, f"CF4_new_128-z008_realization{nsim}_velocity.fits")  # noqa
        else:
            raise ValueError(f"Unknown field kind: `{kind}`.")

        field = fits.open(fpath)[0].data

        # https://projets.ip2i.in2p3.fr//cosmicflows/ says to multiply by 52
        if kind == "velocity":
            field *= 52

        return field.astype(np.float32)
    elif simname == "Lilow2024":
        folder = "/mnt/extraspace/rstiskalek/catalogs"
        warn(f"Using local paths from `{folder}`.", RuntimeWarning)

        if kind == "density":
            fpath = join(folder, "Lilow2024_density.npy")
            field = np.load(fpath)
        elif kind == "velocity":
            field = []
            for p in ["x", "y", "z"]:
                fpath = join(folder, f"Lilow2024_{p}Velocity.npy")
                field.append(np.load(fpath).astype(np.float32))
            field = np.stack(field)

        return field.astype(np.float32)
    else:
        raise ValueError(f"Unknown simulation name: `{simname}`.")

    # Read in the field.
    if kind == "density":
        field = field_reader.density_field(MAS=MAS, grid=grid)
    elif kind == "velocity":
        field = field_reader.velocity_field(MAS=MAS, grid=grid)
    else:
        raise ValueError(f"Unknown field kind: `{kind}`.")

    return field


def combine_from_simulations(catalogue_name, simname, nsims, outfolder,
                             dumpfolder):
    """
    Combine the results from individual simulations into a single file.

    Parameters
    ----------
    catalogue_name : str
        Catalogue name.
    simname : str
        Simulation name.
    nsims : list
        List of IC realisations.
    outfolder : str
        Output folder.
    dumpfolder : str
        Dumping folder where the temporary files are stored.

    Returns
    -------
    None
    """
    fname_out = join(outfolder, f"los_{catalogue_name}_{simname}.hdf5")
    print(f"Combining results from invidivual simulations to `{fname_out}`.")

    if exists(fname_out):
        remove(fname_out)

    for nsim in nsims:
        fname = join(dumpfolder, f"los_{simname}_{nsim}.hdf5")

        with File(fname, 'r') as f, File(fname_out, 'a') as f_out:
            f_out.create_dataset(f"rdist_{nsim}", data=f["rdist"][:])
            f_out.create_dataset(f"density_{nsim}", data=f["density"][:])
            f_out.create_dataset(f"velocity_{nsim}", data=f["velocity"][:])
            f_out.create_dataset(f"rmax_{nsim}", data=f["rmax"][:])

        # Remove the temporary file.
        remove(fname)

    # Remove the dumping folder.
    rmdir(dumpfolder)

    print("Finished combining results.")


###############################################################################
#                       Main interpolating function                           #
###############################################################################

@jit(nopython=True)
def find_index_of_first_nan(y):
    for n in range(1, len(y)):
        if np.isnan(y[n]):
            return n

    return None


def replace_nan_with_last_finite(x, y, apply_decay):
    n = find_index_of_first_nan(y)

    if n is None:
        return y, x[-1]

    y[n:] = y[n-1]
    rmax = x[n-1]

    if apply_decay:
        # Optionally aply 1 / r decay
        y[n:] *= rmax / x[n:]

    return y, rmax


def interpolate_field(pos, simname, nsim, MAS, grid, dump_folder, r,
                      smooth_scales, verbose=False):
    """
    Interpolate the density and velocity fields along the line of sight.

    Parameters
    ----------
    pos : 2-dimensional array
        RA/dec coordinates of the line of sight.
    simname : str
        Simulation name.
    nsim : int
        IC realisation index.
    MAS : str
        Mass assignment scheme.
    grid : int
        Grid resolution.
    dump_folder : str
        Folder where the temporary files are stored.
    r : 1-dimensional array
        Radial spacing.
    smooth_scales : list
        Smoothing scales.

    Returns
    -------
    None
    """
    boxsize = csiborgtools.simname2boxsize(simname)
    fname_out = join(dump_folder, f"los_{simname}_{nsim}.hdf5")

    # First do the density field.
    if verbose:
        print(f"Interpolating density field for IC realisation `{nsim}`.",
              flush=True)
    density = get_field(simname, nsim, "density", MAS, grid)
    rdist, finterp = csiborgtools.field.evaluate_los(
        density, sky_pos=pos, boxsize=boxsize, rdist=r,
        smooth_scales=smooth_scales, verbose=verbose,
        interpolation_method="linear")

    rmax_density = np.full((len(pos), len(smooth_scales)), np.nan)
    for i in range(len(pos)):
        for j in range(len(smooth_scales)):
            y, current_rmax = replace_nan_with_last_finite(rdist, finterp[i, :, j], False)  # noqa
            finterp[i, :, j] = y
            if current_rmax is not None:
                rmax_density[i, j] = current_rmax

    print(f"Writing temporary file `{fname_out}`.")
    with File(fname_out, 'w') as f:
        f.create_dataset("rdist", data=rdist)
        f.create_dataset("density", data=finterp)

    del density, rdist, finterp
    collect()

    if verbose:
        print(f"Interpolating velocity field for IC realisation `{nsim}`.",
              flush=True)
    velocity = get_field(simname, nsim, "velocity", MAS, grid)
    rdist, finterp = csiborgtools.field.evaluate_los(
        velocity[0], velocity[1], velocity[2],
        sky_pos=pos, boxsize=boxsize, rdist=r, smooth_scales=smooth_scales,
        verbose=verbose, interpolation_method="linear")

    rmax_velocity = np.full((3, len(pos), len(smooth_scales)), np.nan)
    for k in range(3):
        for i in range(len(pos)):
            for j in range(len(smooth_scales)):
                y, current_rmax = replace_nan_with_last_finite(rdist, finterp[k][i, :, j], True)  # noqa
                finterp[k][i, :, j] = y
                if current_rmax is not None:
                    rmax_velocity[k, i, j] = current_rmax
    rmax_velocity = np.min(rmax_velocity, axis=0)

    rmax = np.minimum(rmax_density, rmax_velocity)
    with File(fname_out, 'a') as f:
        f.create_dataset("velocity", data=finterp)
        f.create_dataset("rmax", data=rmax)


###############################################################################
#                           Command line interface                            #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--catalogue", type=str, help="Catalogue name.")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. `-1` for all simulations.")
    parser.add_argument("--simname", type=str, help="Simulation name.")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS", "SPH"],
                        help="Mass assignment scheme.")
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    args = parser.parse_args()

    Om0 = csiborgtools.simname2Omega_m(args.simname)
    # r = make_spacing(200, 0.75, 23.25, 34, 0.01, Om0)
    r = np.arange(0, 200, 0.5)

    # smooth_scales = [0, 2, 4, 6, 8]
    smooth_scales = [0, 8]

    print(f"Running catalogue {args.catalogue} for simulation {args.simname} "
          f"with {len(r)} radial points.")

    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_los"
    # Create the dumping folder.
    if comm.Get_rank() == 0:
        dump_folder = join(out_folder,
                           f"temp_{str(datetime.now())}".replace(" ", "_"))
        print(f"Creating folder `{dump_folder}`.")
        makedirs(dump_folder)
    else:
        dump_folder = None
    dump_folder = comm.bcast(dump_folder, root=0)

    # Get the line of sight sky coordinates.
    pos = get_los(args.catalogue, args.simname, comm)

    def main(nsim):
        interpolate_field(pos, args.simname, nsim, args.MAS, args.grid,
                          dump_folder, r, smooth_scales,
                          verbose=comm.Get_size() == 1)

    work_delegation(main, nsims, comm, master_verbose=True)
    comm.Barrier()

    if comm.Get_rank() == 0:
        combine_from_simulations(args.catalogue, args.simname, nsims,
                                 out_folder, dump_folder)
        print("All finished!")
