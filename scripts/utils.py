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

"""Notebook utility funnctions."""

try:
    import galomatch
except ModuleNotFoundError:
    import sys
    sys.path.append("../")


def load_mmain_convert(n):
    srcdir = "/users/hdesmond/Mmain"
    arr = galomatch.io.read_mmain(n, srcdir)

    galomatch.io.convert_mass_cols(arr, "mass_cl")
    galomatch.io.convert_position_cols(arr, ["peak_x", "peak_y", "peak_z"])
    galomatch.io.flip_cols(arr, "peak_x", "peak_z")

    d, ra, dec = galomatch.utils.cartesian_to_radec(arr)
    arr = galomatch.utils.add_columns(arr, [d, ra, dec], ["dist", "ra", "dec"])
    return arr


def load_mmains(N=None, verbose=True):
    from tqdm import tqdm
    ids = galomatch.io.get_csiborg_ids("/mnt/extraspace/hdesmond")
    N = ids.size if N is None else N
    if N > ids.size:
        raise ValueError("`N` cannot be larger than 101.")
    out = [None] * N
    iters = tqdm(range(N)) if verbose else range(N)
    for i in iters:
        out[i] = load_mmain_convert(ids[i])
    return out


def load_planck2015(max_comdist=214):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)
    fpath = ("/mnt/zfsusers/rstiskalek/galomatch/"
             + "data/HFI_PCCS_SZ-union_R2.08.fits")
    return galomatch.io.read_planck2015(fpath, cosmo, max_comdist)


def load_2mpp():
    return galomatch.io.read_2mpp("../data/2M++_galaxy_catalog.dat")
