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
"""
Tools for interpolating 3D fields at arbitrary positions.
"""
import MAS_library as MASL
import numpy as np
import smoothing_library as SL
from numba import jit
from scipy.interpolate import RegularGridInterpolator
from astropy.coordinates import SkyCoord, Supergalactic, Galactic, ICRS
from astropy.coordinates import CartesianRepresentation
from tqdm import tqdm

from ..utils import periodic_wrap_grid, radec_to_cartesian
from .utils import divide_nonzero, force_single_precision, nside2radec

###############################################################################
#                       Cartesian interpolation                               #
###############################################################################


def evaluate_cartesian_cic(*fields, pos, smooth_scales=None, verbose=False):
    """
    Evaluate a scalar field(s) at Cartesian coordinates `pos` using CIC
    interpolation.

    Parameters
    ----------
    field : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Fields to be interpolated.
    pos : 2-dimensional array of shape `(n_samples, 3)`
        Query positions in box units.
    smooth_scales : (list of) float, optional
        Smoothing scales in box units. If `None`, no smoothing is performed.
    verbose : bool, optional
        Smoothing verbosity flag.

    Returns
    -------
    (list of) 2-dimensional array of shape `(n_samples, len(smooth_scales))`
    """
    pos = force_single_precision(pos)

    if isinstance(smooth_scales, (int, float)):
        smooth_scales = [smooth_scales]

    if smooth_scales is None:
        shape = (pos.shape[0],)
    else:
        shape = (pos.shape[0], len(smooth_scales))

    interp_fields = [np.full(shape, np.nan, dtype=np.float32)
                     for __ in range(len(fields))]

    for i, field in enumerate(fields):
        if smooth_scales is None:
            MASL.CIC_interp(field, 1., pos, interp_fields[i])
        else:
            desc = f"Smoothing and interpolating field {i + 1}/{len(fields)}"
            iterator = tqdm(smooth_scales, desc=desc, disable=not verbose)

            for j, scale in enumerate(iterator):
                if not scale > 0:
                    fsmooth = np.copy(field)
                else:
                    fsmooth = smoothen_field(field, scale, 1., make_copy=True)
                MASL.CIC_interp(fsmooth, 1., pos, interp_fields[i][:, j])

    if len(fields) == 1:
        return interp_fields[0]

    return interp_fields


def evaluate_cartesian_regular(*fields, pos, smooth_scales=None,
                               method="linear", verbose=False):
    """
    Evaluate a scalar field(s) at Cartesian coordinates `pos` using linear
    interpolation on a regular grid.

    Parameters
    ----------
    *fields : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Fields to be interpolated.
    pos : 2-dimensional array of shape `(n_samples, 3)`
        Query positions in box units.
    smooth_scales : (list of) float, optional
        Smoothing scales in box units. If `None`, no smoothing is performed.
    method : str, optional
        Interpolation method, must be one of the methods of
        `scipy.interpolate.RegularGridInterpolator`.
    verbose : bool, optional
        Smoothing verbosity flag.

    Returns
    -------
    (list of) 2-dimensional array of shape `(n_samples, len(smooth_scales))`
    """
    pos = force_single_precision(pos)

    if isinstance(smooth_scales, (int, float)):
        smooth_scales = [smooth_scales]

    if smooth_scales is None:
        shape = (pos.shape[0],)
    else:
        shape = (pos.shape[0], len(smooth_scales))

    ngrid = fields[0].shape[0]
    cellsize = 1. / ngrid

    X = np.linspace(0.5 * cellsize, 1 - 0.5 * cellsize, ngrid)
    Y, Z = np.copy(X), np.copy(X)

    interp_fields = [np.full(shape, np.nan, dtype=np.float32)
                     for __ in range(len(fields))]
    for i, field in enumerate(fields):
        if smooth_scales is None:
            field_interp = RegularGridInterpolator(
                (X, Y, Z), field, fill_value=np.nan, bounds_error=False,
                method=method)
            interp_fields[i] = field_interp(pos)
        else:
            desc = f"Smoothing and interpolating field {i + 1}/{len(fields)}"
            iterator = tqdm(smooth_scales, desc=desc, disable=not verbose)

            for j, scale in enumerate(iterator):
                if not scale > 0:
                    fsmooth = np.copy(field)
                else:
                    fsmooth = smoothen_field(field, scale, 1., make_copy=True)

                field_interp = RegularGridInterpolator(
                    (X, Y, Z), fsmooth, fill_value=np.nan, bounds_error=False,
                    method=method)
                interp_fields[i][:, j] = field_interp(pos)

    if len(fields) == 1:
        return interp_fields[0]

    return interp_fields


def observer_peculiar_velocity(velocity_field, smooth_scales=None,
                               observer=None, verbose=True):
    """
    Calculate the peculiar velocity in the centre of the box.

    Parameters
    ----------
    velocity_field : 4-dimensional array of shape `(3, grid, grid, grid)`
        Velocity field in `km / s`.
    smooth_scales : (list of) float, optional
        Smoothing scales in box units. If `None`, no smoothing is performed.
    observer : 1-dimensional array of shape `(3,)`, optional
        Observer position in box units. If `None`, the observer is assumed to
        be in the centre of the box.
    verbose : bool, optional
        Smoothing verbosity flag.

    Returns
    -------
    vpec : 1-dimensional array of shape `(3,)` or `(len(smooth_scales), 3)`
    """
    if observer is None:
        pos = np.asanyarray([0.5, 0.5, 0.5]).reshape(1, 3)
    else:
        pos = np.asanyarray(observer).reshape(1, 3)

    vx, vy, vz = evaluate_cartesian_cic(
        *velocity_field, pos=pos, smooth_scales=smooth_scales, verbose=verbose)

    # Reshape since we evaluated only one point
    vx = vx.reshape(-1, )
    vy = vy.reshape(-1, )
    vz = vz.reshape(-1, )

    if smooth_scales is None:
        return np.array([vx[0], vy[0], vz[0]])

    return np.vstack([vx, vy, vz]).T

###############################################################################
#                   Evaluating the fields along a LOS                         #
###############################################################################


def evaluate_los(*fields, sky_pos, boxsize, rmax, dr, smooth_scales=None,
                 interpolation_method="cic", verbose=False):
    """
    Interpolate the fields for a set of lines of sights from the observer
    in the centre of the box.

    Parameters
    ----------
    *fields : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Fields to be interpolated.
    sky_pos : 2-dimensional array of shape `(n_samples, 2)`
        Query positions in spherical coordinates (RA, dec) in degrees.
    boxsize : float
        Box size in `Mpc / h`.
    rmax : float
        Maximum radial distance in `Mpc / h`.
    dr : float
        Radial distance step in `Mpc / h`.
    smooth_scales : (list of) float, optional
        Smoothing scales in `Mpc / h`.
    interpolation_method : str, optional
        Interpolation method. Must be one of `cic` or one of the methods of
        `scipy.interpolate.RegularGridInterpolator`.
    verbose : bool, optional
        Smoothing verbosity flag.

    Returns
    -------
    rdist : 1-dimensional array
        Radial positions in `Mpc / h` where the fields were evaluated.
    field_interp : (list of) 2- or 3-dimensional arrays of shape `(n_query, len(rdist), len(smooth_scales))`  # noqa
        The interpolated fields. If `smooth_scales` is `None`, the last
        is omitted.
    """
    mpc2box = 1. / boxsize

    if not isinstance(sky_pos, np.ndarray) and sky_pos.ndim != 2:
        raise ValueError("`sky_pos` must be a 2D array.")
    nsamples = len(sky_pos)

    if interpolation_method == "cic" and rmax > 0.5 * boxsize:
        raise ValueError("The maximum radius must be within the box.")

    # Radial positions to evaluate for each line of sight.
    rdist = np.arange(0, rmax, dr, dtype=fields[0].dtype)

    # Create an array of radial positions and sky coordinates of each line of
    # sight.
    pos = np.empty((nsamples * len(rdist), 3), dtype=fields[0].dtype)
    for i in range(nsamples):
        start, end = i * len(rdist), (i + 1) * len(rdist)
        pos[start:end, 0] = rdist * mpc2box
        pos[start:end, 1] = sky_pos[i, 0]
        pos[start:end, 2] = sky_pos[i, 1]

    pos = force_single_precision(pos)
    # Convert the spherical coordinates to Cartesian coordinates.
    pos = radec_to_cartesian(pos) + 0.5

    if smooth_scales is not None:
        if isinstance(smooth_scales, (int, float)):
            smooth_scales = [smooth_scales]

        if isinstance(smooth_scales, list):
            smooth_scales = np.array(smooth_scales, dtype=np.float32)

        smooth_scales *= mpc2box

    if interpolation_method == "cic":
        field_interp = evaluate_cartesian_cic(
            *fields, pos=pos, smooth_scales=smooth_scales,
            verbose=verbose)
    else:
        field_interp = evaluate_cartesian_regular(
            *fields, pos=pos, smooth_scales=smooth_scales,
            method=interpolation_method, verbose=verbose)

    if len(fields) == 1:
        field_interp = [field_interp]

    # Now we reshape the interpolated field to have the same shape as the
    # input `sky_pos`.
    if smooth_scales is None:
        shape_single = (nsamples, len(rdist))
    else:
        shape_single = (nsamples, len(rdist), len(smooth_scales))

    field_interp_reshaped = [None] * len(fields)
    for i in range(len(fields)):
        samples = np.full(shape_single, np.nan, dtype=field_interp[i].dtype)
        for j in range(nsamples):
            start, end = j * len(rdist), (j + 1) * len(rdist)
            samples[j] = field_interp[i][start:end, ...]

        field_interp_reshaped[i] = samples

    if len(fields) == 1:
        return rdist, field_interp_reshaped[0]

    return rdist, field_interp_reshaped


###############################################################################
#                              Sky maps                                       #
###############################################################################


def make_sky(field, angpos, rmax, dr, boxsize, interpolation_method="cic",
             return_full=False, verbose=True):
    r"""
    Make a sky map of a scalar field. The observer is in the centre of the
    box the field is evaluated along directions `angpos` (RA [0, 360) deg,
    dec [-90, 90] deg). The field is evaluated up to `rmax` with a linear
    spacing of `dr` in `Mpc / h`.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Field to be interpolated
    angpos : 2-dimensional arrays of shape `(ndir, 2)`
        Directions to evaluate the field.
    rmax : float
        Maximum radial distance in `Mpc / h`.
    dr : float
        Radial distance step in `Mpc / h`.
    boxsize : float
        Box size in `Mpc / h`.
    interpolation_method : str, optional
        Interpolation method. Must be one of `cic` or one of the methods of
        `scipy.interpolate.RegularGridInterpolator`.
    return_full : bool, optional
        If `True`, return the full interpolated field instead of the average
        field at each radial distance.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    interp_field : 1-dimensional array of shape `(n_pos, )`
    """
    rdist, finterp = evaluate_los(
        field, sky_pos=angpos, boxsize=boxsize, rmax=rmax, dr=dr,
        smooth_scales=None, verbose=verbose,
        interpolation_method=interpolation_method)

    if return_full:
        return rdist, finterp

    finterp *= rdist**2
    finterp = np.trapz(finterp, x=rdist, axis=1)
    finterp /= np.trapz(rdist**2, x=rdist)

    return finterp


###############################################################################
#                       Supergalactic plane slice                             #
###############################################################################


def xy_supergalactic_slice(field, boxsize, xmin, xmax, ngrid, field_frame,
                           z_value=0):
    """
    Create a 2D slice of a scalar field in the x-y supergalactic plane.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Field to be interpolated.
    boxsize : float
        Box size in `Mpc / h`.
    xmin, xmax : float
        Minimum and maximum x and y values in supergalactic coordinates.
    ngrid : int
        Number of grid points along each axis.
    field_frame : str
        Frame of the field. Must be one of `galactic`, `supergalactic` or
        `icrs`.
    z_value : float, optional
        Value of the z-coordinate in supergalactic coordinates.

    Returns
    -------
    2-dimensional array of shape `(ngrid, ngrid)`
    """
    # Coordinates of the 2D slice in supergalactic coordinates
    xgrid = np.linspace(xmin, xmax, ngrid)
    ygrid = np.copy(xgrid)
    grid = np.stack(np.meshgrid(xgrid, ygrid))

    grid = grid.reshape(2, -1).T
    grid = np.hstack([grid, np.ones(ngrid**2).reshape(-1, 1) * z_value])

    supergalactic_coord = SkyCoord(CartesianRepresentation(
        grid[:, 0], grid[:, 1], grid[:, 2]), frame=Supergalactic)

    # Create a Supergalactic SkyCoord object from Cartesian coordinates
    if field_frame == "galactic":
        original_frame = Galactic
    elif field_frame == "supergalactic":
        original_frame = Supergalactic
    elif field_frame == "icrs":
        original_frame = ICRS
    else:
        raise ValueError(f"Unknown field frame: {field_frame}")

    # Convert to field frame
    coords = supergalactic_coord.transform_to(original_frame).cartesian
    pos = np.stack([coords.x, coords.y, coords.z]).value
    pos = pos.T

    # Convert to appropriate box units
    pos /= boxsize
    pos += 0.5

    if np.any(pos <= 0) or np.any(pos >= 1):
        raise ValueError("Some positions are outside the box.")

    return evaluate_cartesian_cic(field, pos=pos).reshape(ngrid, ngrid)


###############################################################################
#                     Average field at a radial distance                      #
###############################################################################


def field_at_distance(field, distance, boxsize, smooth_scales=None, nside=128,
                      verbose=True):
    """
    Evaluate a scalar field at uniformly spaced angular coordinates at a
    given distance from the observer

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Field to be interpolated.
    distance : float
        Distance from the observer in `Mpc / h`.
    boxsize : float
        Box size in `Mpc / h`.
    smooth_scales : (list of) float, optional
        Smoothing scales in `Mpc / h`. If `None`, no smoothing is performed.
    nside : int, optional
        HEALPix nside. Used to generate the uniformly spaced angular
        coordinates. Recommended to be >> 1.
    verbose : bool, optional
        Smoothing verbosity flag.

    Returns
    -------
    vals : n-dimensional array of shape `(npix, len(smooth_scales))`
    """
    # Get positions of HEALPix pixels on the sky and then convert those to
    # box Cartesian coordinates. We take HEALPix pixels because they are
    # uniformly distributed on the sky.
    angpos = nside2radec(nside)
    X = np.hstack([np.ones(len(angpos)).reshape(-1, 1) * distance, angpos])
    X = radec_to_cartesian(X) / boxsize + 0.5

    return evaluate_cartesian_cic(field, pos=X, smooth_scales=smooth_scales,
                                  verbose=verbose)


###############################################################################
#                     Real-to-redshift space field dragging                   #
###############################################################################


@jit(nopython=True)
def make_gridpos(grid_size):
    """Make a regular grid of positions and distances from the center."""
    grid_pos = np.full((grid_size**3, 3), np.nan, dtype=np.float32)
    grid_dist = np.full(grid_size**3, np.nan, dtype=np.float32)

    n = 0
    for i in range(grid_size):
        px = (i - 0.5 * (grid_size - 1)) / grid_size
        px2 = px**2
        for j in range(grid_size):
            py = (j - 0.5 * (grid_size - 1)) / grid_size
            py2 = py**2
            for k in range(grid_size):
                pz = (k - 0.5 * (grid_size - 1)) / grid_size
                pz2 = pz**2

                grid_pos[n, 0] = px
                grid_pos[n, 1] = py
                grid_pos[n, 2] = pz

                grid_dist[n] = (px2 + py2 + pz2)**0.5

                n += 1

    return grid_pos, grid_dist


def field2rsp(field, radvel_field, box, MAS, init_value=0.):
    """
    Forward model a real space scalar field to redshift space.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Real space field to be evolved to redshift space.
    radvel_field : 3-dimensional array of shape `(grid, grid, grid)`
        Radial velocity field in `km / s`. Expected to account for the observer
        velocity.
    box : :py:class:`csiborgtools.read.CSiBORG1Box`
        The simulation box information and transformations.
    MAS : str
        Mass assignment. Must be one of `NGP`, `CIC`, `TSC` or `PCS`.
    init_value : float, optional
        Initial value of the RSP field on the grid.

    Returns
    -------
    3-dimensional array of shape `(grid, grid, grid)`
    """
    grid = field.shape[0]
    H0_inv = 1. / 100 / box.box2mpc(1.)

    # Calculate the regular grid positions and distances from the center.
    grid_pos, grid_dist = make_gridpos(grid)
    grid_dist = grid_dist.reshape(-1, 1)

    # Move the grid positions to redshift space.
    grid_pos *= (1 + H0_inv * radvel_field.reshape(-1, 1) / grid_dist)
    grid_pos += 0.5
    grid_pos = periodic_wrap_grid(grid_pos)

    rsp_field = np.full(field.shape, init_value, dtype=np.float32)
    cell_counts = np.zeros(rsp_field.shape, dtype=np.float32)

    # Interpolate the field to the grid positions.
    MASL.MA(grid_pos, rsp_field, 1., MAS, W=field.reshape(-1,))
    MASL.MA(grid_pos, cell_counts, 1., MAS)
    divide_nonzero(rsp_field, cell_counts)

    return rsp_field

###############################################################################
#                          Supplementary function                             #
###############################################################################


@jit(nopython=True)
def fill_outside(field, fill_value, rmax, boxsize):
    """
    Fill cells outside of a sphere of radius `rmax` around the box centre with
    `fill_value`.
    """
    imax, jmax, kmax = field.shape
    assert imax == jmax == kmax
    N = imax

    # Squared radial distance from the center of the box in box units.
    rmax_box2 = (N * rmax / boxsize)**2

    for i in range(N):
        idist2 = (i - 0.5 * (N - 1))**2
        for j in range(N):
            jdist2 = (j - 0.5 * (N - 1))**2
            for k in range(N):
                kdist2 = (k - 0.5 * (N - 1))**2
                if idist2 + jdist2 + kdist2 > rmax_box2:
                    field[i, j, k] = fill_value
    return field


def smoothen_field(field, smooth_scale, boxsize, threads=1, make_copy=False):
    """
    Smooth a field with a Gaussian filter.
    """
    W_k = SL.FT_filter(boxsize, smooth_scale, field.shape[0], "Gaussian",
                       threads)

    if make_copy:
        field = np.copy(field)

    return SL.field_smoothing(field, W_k, threads)
