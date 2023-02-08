# Copyright (C) 2022 Richard Stiskalek, Harry Desmond
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
Functions to read in the particle and clump files.
"""
import numpy
from os.path import join
from sklearn.neighbors import NearestNeighbors
from .readsim import (CSiBORGPaths, read_mmain, read_initcm)
from ..utils import (flip_cols, add_columns)
from ..units import (BoxUnits, cartesian_to_radec)


class HaloCatalogue:
    r"""
    Processed halo catalogue, the data should be calculated in `run_fit_halos`.

    Parameters
    ----------
    paths : py:class:`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths-handling object with set `n_sim` and `n_snap`.
    min_m500 : float, optional
        The minimum :math:`M_{rm 500c} / M_\odot` mass. By default no
        threshold.
    max_dist : float, optional
        The maximum comoving distance of a halo. By default no upper limit.
    """
    _box = None
    _paths = None
    _data = None
    _knn = None
    _knn0 = None
    _positions = None
    _positions0 = None

    def __init__(self, nsim, min_m500=None, max_dist=None):
        # Set up paths
        paths = CSiBORGPaths(n_sim=nsim)
        paths.n_snap = paths.get_maximum_snapshot()
        self._paths = paths
        self._box = BoxUnits(paths)
        min_m500 = 0 if min_m500 is None else min_m500
        max_dist = numpy.infty if max_dist is None else max_dist
        self._paths = paths
        self._set_data(min_m500, max_dist)
        # Initialise the KNN at z = 0 and at z = 70
        knn = NearestNeighbors()
        knn.fit(self.positions)
        self._knn = knn

        knn0 = NearestNeighbors()
        knn0.fit(self.positions0)
        self._knn0 = knn0

    @property
    def data(self):
        """
        Halo catalogue.

        Returns
        -------
        cat : structured array
        """
        if self._data is None:
            raise ValueError("`data` is not set!")
        return self._data

    @property
    def box(self):
        """
        Box object, useful for change of units.

        Returns
        -------
        box : :py:class:`csiborgtools.units.BoxUnits`
        """
        return self._box

    @property
    def cosmo(self):
        """
        The box cosmology.

        Returns
        -------
        cosmo : `astropy` cosmology object
        """
        return self.box.cosmo

    @property
    def paths(self):
        """
        The paths-handling object.

        Returns
        -------
        paths : :py:class:`csiborgtools.read.CSiBORGPaths`
        """
        return self._paths

    @property
    def n_snap(self):
        """
        The snapshot ID.

        Returns
        -------
        n_snap : int
        """
        return self.paths.n_snap

    @property
    def n_sim(self):
        """
        The initial condition (IC) realisation ID.

        Returns
        -------
        n_sim : int
        """
        return self.paths.n_sim

    @property
    def knn(self):
        """
        The final snapshot k-nearest neighbour object.

        Returns
        -------
        knn : :py:class:`sklearn.neighbors.NearestNeighbors`
        """
        return self._knn

    @property
    def knn0(self):
        """
        The initial snapshot k-nearest neighbour object.

        Returns
        -------
        knn : :py:class:`sklearn.neighbors.NearestNeighbors`
        """
        return self._knn0

    def _set_data(self, min_m500, max_dist):
        """
        Loads the data, merges with mmain, does various coordinate transforms.
        """
        # Load the processed data
        fname = "ramses_out_{}_{}.npy".format(
            str(self.n_sim).zfill(5), str(self.n_snap).zfill(5))
        data = numpy.load(join(self.paths.dumpdir, fname))

        # Load the mmain file and add it to the data
        mmain = read_mmain(self.n_sim, self.paths.mmain_path)
        data = self.merge_mmain_to_clumps(data, mmain)
        flip_cols(data, "peak_x", "peak_z")

        # Cut on number of particles and finite m200
        data = data[(data["npart"] > 100) & numpy.isfinite(data["m200"])]

        # Now also load the initial positions
        initcm = read_initcm(self.n_sim, self.paths.initmatch_path)
        if initcm is not None:
            data = self.merge_initmatch_to_clumps(data, initcm)
            flip_cols(data, "x0", "z0")

        # Calculate redshift
        pos = [data["peak_{}".format(p)] - 0.5 for p in ("x", "y", "z")]
        vel = [data["v{}".format(p)] for p in ("x", "y", "z")]
        zpec = self.box.box2pecredshift(*vel, *pos)
        zobs = self.box.box2obsredshift(*vel, *pos)
        zcosmo = self.box.box2cosmoredshift(
            sum(pos[i]**2 for i in range(3))**0.5)

        data = add_columns(data, [zpec, zobs, zcosmo],
                           ["zpec", "zobs", "zcosmo"])

        # Unit conversion
        convert_cols = ["m200", "m500", "totpartmass", "mass_mmain",
                        "r200", "r500", "Rs", "rho0",
                        "peak_x", "peak_y", "peak_z"]
        data = self.box.convert_from_boxunits(data, convert_cols)

        # Cut on mass. Note that this is in Msun
        data = data[data["m500"] > min_m500]

        # Now calculate spherical coordinates
        d, ra, dec = cartesian_to_radec(
            data["peak_x"], data["peak_y"], data["peak_z"])

        data = add_columns(data, [d, ra, dec], ["dist", "ra", "dec"])

        # Cut on separation
        data = data[data["dist"] < max_dist]

        # Pre-allocate the positions arrays
        self._positions = numpy.vstack(
            [data["peak_{}".format(p)] for p in ("x", "y", "z")]).T
        self._positions = self._positions.astype(numpy.float32)
        # And do the unit transform
        if initcm is not None:
            data = self.box.convert_from_boxunits(
                data, ["x0", "y0", "z0", "lagpatch"])
            self._positions0 = numpy.vstack(
                [data["{}0".format(p)] for p in ("x", "y", "z")]).T
            self._positions0 = self._positions0.astype(numpy.float32)

        # Convert all that is not an integer to float32
        names = list(data.dtype.names)
        formats = []
        for name in names:
            if data[name].dtype.char in numpy.typecodes["AllInteger"]:
                formats.append(numpy.int32)
            else:
                formats.append(numpy.float32)
        dtype = numpy.dtype({"names": names, "formats": formats})
        data = data.astype(dtype)

        self._data = data

    def merge_mmain_to_clumps(self, clumps, mmain):
        """
        Merge columns from the `mmain` files to the `clump` file, matches them
        by their halo index while assuming that the indices `index` in both
        arrays are sorted.

        Parameters
        ----------
        clumps : structured array
            Clumps structured array.
        mmain : structured array
            Parent halo array whose information is to be merged into `clumps`.

        Returns
        -------
        out : structured array
            Array with added columns.
        """
        X = numpy.full((clumps.size, 2), numpy.nan)
        # Mask of which clumps have a mmain index
        mask = numpy.isin(clumps["index"], mmain["index"])

        X[mask, 0] = mmain["mass_cl"]
        X[mask, 1] = mmain["sub_frac"]
        return add_columns(clumps, X, ["mass_mmain", "sub_frac"])

    def merge_initmatch_to_clumps(self, clumps, initcat):
        """
        Merge columns from the `init_cm` files to the `clump` file.

        Parameters
        ----------
        clumps : structured array
            Clumps structured array.
        initcat : structured array
            Catalog with the clumps initial centre of mass at z = 70.

        Returns
        -------
        out : structured array
        """
        # There are more initcat clumps, so check which ones have z = 0
        # and then downsample
        mask = numpy.isin(initcat["ID"], clumps["index"])
        initcat = initcat[mask]
        # Now the index ordering should match
        if not numpy.alltrue(initcat["ID"] == clumps["index"]):
            raise ValueError(
                "Ordering of `initcat` and `clumps` is inconsistent.")

        X = numpy.full((clumps.size, 4), numpy.nan)
        for i, p in enumerate(['x', 'y', 'z', "lagpatch"]):
            X[:, i] = initcat[p]
        return add_columns(clumps, X, ["x0", "y0", "z0", "lagpatch"])

    @property
    def positions(self):
        """
        3D positions of halos in comoving units of Mpc.

        Returns
        -------
        X : 2-dimensional array
            Array of shape `(n_halos, 3)`, where the latter axis represents
            `x`, `y` and `z`.
        """
        return self._positions

    @property
    def positions0(self):
        r"""
        3D positions of halos in the initial snapshot in comoving units of Mpc.

        Returns
        -------
        X : 2-dimensional array
            Array of shape `(n_halos, 3)`, where the latter axis represents
            `x`, `y` and `z`.
        """
        if self._positions0 is None:
            raise RuntimeError("Initial positions are not set!")
        return self._positions0

    @property
    def velocities(self):
        """
        Cartesian velocities of halos.

        Returns
        -------
        vel : 2-dimensional array
            Array of shape `(n_halos, 3)`.
        """
        return numpy.vstack([self["v{}".format(p)] for p in ("x", "y", "z")]).T

    @property
    def angmomentum(self):
        """
        Angular momentum of halos in the box coordinate system.

        Returns
        -------
        angmom : 2-dimensional array
            Array of shape `(n_halos, 3)`.
        """
        return numpy.vstack([self["L{}".format(p)] for p in ("x", "y", "z")]).T

    def radius_neigbours(self, X, radius, select_initial=True):
        r"""
        Return sorted nearest neigbours within `radius` of `X` in the initial
        or final snapshot.

        Parameters
        ----------
        X : 2-dimensional array
            Array of shape `(n_queries, 3)`, where the latter axis represents
            `x`, `y` and `z`.
        radius : float
            Limiting distance of neighbours.
        select_initial : bool, optional
            Whether to search for neighbours in the initial or final snapshot.
            By default `True`, i.e. the final snapshot.

        Returns
        -------
        dist : list of 1-dimensional arrays
            List of length `n_queries` whose elements are arrays of distances
            to the nearest neighbours.
        knns : list of 1-dimensional arrays
            List of length `n_queries` whose elements are arrays of indices of
            nearest neighbours in this catalogue.
        """
        if not (X.ndim == 2 and X.shape[1] == 3):
            raise TypeError("`X` must be an array of shape `(n_samples, 3)`.")
        knn = self.knn0 if select_initial else self.knn  # Pick the right KNN
        return knn.radius_neighbors(X, radius, sort_results=True)

    @property
    def keys(self):
        """Catalogue keys."""
        return self.data.dtype.names

    def __getitem__(self, key):
        initpars = ["x0", "y0", "z0"]
        if key in initpars and key not in self.keys:
            raise RuntimeError("Initial positions are not set!")
        return self._data[key]


def concatenate_clumps(clumps):
    """
    Concatenate an array of clumps to a single array containing all particles.

    Parameters
    ----------
    clumps : list of structured arrays

    Returns
    -------
    particles : structured array
    """
    # Count how large array will be needed
    N = 0
    for clump, __ in clumps:
        N += clump.size
    # Infer dtype of positions
    if clumps[0][0]['x'].dtype.char in numpy.typecodes["AllInteger"]:
        posdtype = numpy.int32
    else:
        posdtype = numpy.float32

    # Pre-allocate array
    dtype = {"names": ['x', 'y', 'z', 'M'],
             "formats": [posdtype] * 3 + [numpy.float32]}
    particles = numpy.full(N, numpy.nan, dtype)

    # Fill it one clump by another
    start = 0
    for clump, __ in clumps:
        end = start + clump.size
        for p in ('x', 'y', 'z', 'M'):
            particles[p][start:end] = clump[p]
        start = end

    return particles


def clumps_pos2cell(clumps, overlapper):
    """
    Convert clump positions directly to cell IDs. Useful to speed up subsequent
    calculations. Overwrites the passed in arrays.

    Parameters
    ----------
    clumps : array of arrays
        Array of clump structured arrays whose `x`, `y`, `z` keys will be
        converted.
    overlapper : py:class:`csiborgtools.match.ParticleOverlapper`
        `ParticleOverlapper` handling the cell assignment.

    Returns
    -------
    None
    """
    # Check if clumps are probably already in cells
    if any(clumps[0][0].dtype[p].char in numpy.typecodes["AllInteger"]
           for p in ('x', 'y', 'z')):
        raise ValueError("Positions appear to already be converted cells.")

    # Get the new dtype that replaces float for int for positions
    names = clumps[0][0].dtype.names  # Take the first one, doesn't matter
    formats = [descr[1] for descr in clumps[0][0].dtype.descr]

    for i in range(len(names)):
        if names[i] in ('x', 'y', 'z'):
            formats[i] = numpy.int32
    dtype = numpy.dtype({"names": names, "formats": formats})

    # Loop switch positions for cells IDs and change dtype
    for n in range(clumps.size):
        for p in ('x', 'y', 'z'):
            clumps[n][0][p] = overlapper.pos2cell(clumps[n][0][p])
        clumps[n][0] = clumps[n][0].astype(dtype)
