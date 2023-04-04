# Copyright (C) 2022 Richard Stiskalek, Deaglan Bartlett
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
Tools for splitting the particles and a clump object.
"""
from os import remove
from warnings import warn
from os.path import join
import numpy
from tqdm import trange
from ..read import ParticleReader


def clump_with_particles(particle_clumps, clumps):
    """
    Count how many particles does each clump have.

    Parameters
    ----------
    particle_clumps : 1-dimensional array
        Array of particles' clump IDs.
    clumps : structured array
        Clumps array.

    Returns
    -------
    with_particles : 1-dimensional boolean array
        Array of whether a clump has any particles.
    """
    return numpy.isin(clumps["index"], particle_clumps)


def distribute_halos(n_splits, clumps):
    """
    Evenly distribute clump indices to smaller splits. Clumps should only be
    clumps that contain particles.

    Parameters
    ----------
    n_splits : int
        Number of splits.
    clumps : structured array
        Clumps array.

    Returns
    -------
    splits : 2-dimensional array of shape `(njobs, 2)`
        Array of starting and ending indices of each CPU.
    """
    # Make sure these are unique IDs
    indxs = clumps["index"]
    if indxs.size > numpy.unique((indxs)).size:
        raise ValueError("`clump_indxs` constains duplicate indices.")
    Ntotal = indxs.size
    njobs_per_cpu = numpy.ones(n_splits, dtype=int) * Ntotal // n_splits
    # Split the remainder Ntotal % njobs among the CPU
    njobs_per_cpu[:Ntotal % n_splits] += 1
    start = ParticleReader.nparts_to_start_ind(njobs_per_cpu)
    return numpy.vstack([start, start + njobs_per_cpu]).T


def dump_split_particles(particles, particle_clumps, clumps, n_splits,
                         nsnap, nsim, paths, verbose=True):
    """
    Save the data needed for each split so that a process does not have to load
    everything.

    Parameters
    ----------
    particles : structured array
        The particle array.
    particle_clumps : 1-dimensional array
        Array of particles' clump IDs.
    clumps : structured array
        The clumps array.
    n_splits : int
        Number of times to split the clumps.
    nsnap : int
        Snapshot index.
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths-handling object with set `n_sim` and `n_snap`.
    verbose : bool, optional
        Verbosity flag. By default `True`.

    Returns
    -------
    None
    """
    if particles.size != particle_clumps.size:
        raise ValueError("`particles` must correspond to `particle_clumps`.")
    # Calculate which clumps have particles
    with_particles = clump_with_particles(particle_clumps, clumps)
    clumps = numpy.copy(clumps)[with_particles]
    if verbose:
        warn(r"There are {:.4f}% clumps that have identified particles."
             .format(with_particles.sum() / with_particles.size * 100))

    # The starting clump index of each split
    splits = distribute_halos(n_splits, clumps)
    fname = join(paths.temp_dumpdir, "out_{}_snap_{}_{}.npz")

    tot = 0
    for n in trange(n_splits) if verbose else range(n_splits):
        # Lower and upper array index of the clumps array
        i, j = splits[n, :]
        # Clump indices in this split
        indxs = clumps["index"][i:j]
        hmin, hmax = indxs.min(), indxs.max()
        mask = (particle_clumps >= hmin) & (particle_clumps <= hmax)
        # Check number of clumps
        npart_unique = numpy.unique(particle_clumps[mask]).size
        if indxs.size > npart_unique:
            raise RuntimeError(
                "Split `{}` contains more unique clumps (`{}`) than there are "
                "unique particles' clump indices (`{}`)after removing clumps "
                "with no particles.".format(n, indxs.size, npart_unique))
        # Dump it!
        tot += mask.sum()
        fout = fname.format(nsim, nsnap, n)
        numpy.savez(fout, particles[mask], particle_clumps[mask], clumps[i:j])

    # There are particles whose clump ID is > 1 and have no counterpart in the
    # clump file. Therefore can save fewer particles, depending on the cut.
    if tot > particle_clumps.size:
        raise RuntimeError(
            "Num. of dumped particles `{}` is greater than the particle file "
            "size `{}`.".format(tot, particle_clumps.size))


def split_jobs(njobs, ncpu):
    """
    Split `njobs` amongst `ncpu`.

    Parameters
    ----------
    njobs : int
        Number of jobs.
    ncpu : int
        Number of CPUs.

    Returns
    -------
    jobs : list of lists of integers
        Outer list of each CPU and inner lists for CPU's jobs.
    """
    njobs_per_cpu, njobs_remainder = divmod(njobs, ncpu)
    jobs = numpy.arange(njobs_per_cpu * ncpu).reshape((njobs_per_cpu, ncpu)).T
    jobs = jobs.tolist()
    for i in range(njobs_remainder):
        jobs[i].append(njobs_per_cpu * ncpu + i)

    return jobs


def load_split_particles(nsplit, nsnap, nsim, paths, remove_split=False):
    """
    Load particles of a split saved by `dump_split_particles`.

    Parameters
    --------
    n_split : int
        Split index.
    nsnap : int
        Snapshot index.
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths-handling object with set `n_sim` and `n_snap`.
    remove_split : bool, optional
        Whether to remove the split file. By default `False`.

    Returns
    -------
    particles : structured array
        Particle array of this split.
    clumps_indxs : 1-dimensional array
        Array of particles' clump IDs of this split.
    clumps : 1-dimensional array
        Clumps belonging to this split.
    """
    fname = join(paths.temp_dumpdir,
                 "out_{}_snap_{}_{}.npz".format(nsim, nsnap, nsplit))
    file = numpy.load(fname)
    particles, clump_indxs, clumps = (file[f] for f in file.files)
    if remove_split:
        remove(fname)
    return particles, clump_indxs, clumps


def pick_single_clump(n, particles, particle_clumps, clumps):
    """
    Get particles belonging to the `n`th clump in `clumps` arrays.

    Parameters
    ----------
    n : int
        Clump position in `clumps` array. Not its halo finder index!
    particles : structured array
        Particle array.
    particle_clumps : 1-dimensional array
        Array of particles' clump IDs.
    clumps : structured array
        Array of clumps.

    Returns
    -------
    sel_particles : structured array
        Particles belonging to the requested clump.
    sel_clump : array
        A slice of a `clumps` array corresponding to this clump. Must
        contain `["peak_x", "peak_y", "peak_z", "mass_cl"]`.
    """
    # Clump index on the nth position
    k = clumps["index"][n]
    # Mask of which particles belong to this clump
    mask = particle_clumps == k
    return particles[mask], clumps[n]


###############################################################################
#                           Clump object                                      #
###############################################################################


class Clump:
    r"""
    A clump (halo) object to handle the particles and their clump's data.

    Parameters
    ----------
    x : 1-dimensional array
        Particle coordinates along the x-axis.
    y : 1-dimensional array
        Particle coordinates along the y-axis.
    z : 1-dimensional array
        Particle coordinates along the z-axis.
    m : 1-dimensional array
        Particle masses.
    x0 : float
        Clump center coordinate along the x-axis.
    y0 : float
        Clump center coordinate along the y-axis.
    z0 : float
        Clump center coordinate along the z-axis.
    clump_mass : float, optional
        Mass of the clump. By default not set.
    vx : 1-dimensional array, optional
        Particle velocity along the x-axis. By default not set.
    vy : 1-dimensional array, optional
        Particle velocity along the y-axis. By default not set.
    vz : 1-dimensional array, optional
        Particle velocity along the z-axis. By default not set.
    index : int, optional
        The halo finder index of this clump. By default not set.
    rhoc : float, optional
        The critical density :math:`\rho_c` at this snapshot in box units. By
        default not set.
    G : float, optional
        The gravitational constant :math:`G` in box units. By default not set.
    """
    _pos = None
    _clump_pos = None
    _clump_mass = None
    _vel = None
    _index = None
    _rhoc = None
    _G = None

    def __init__(self, x, y, z, m, x0, y0, z0, clump_mass=None,
                 vx=None, vy=None, vz=None, index=None, rhoc=None, G=None):
        self._pos = numpy.vstack([x - x0, y - y0, z - z0]).T
        self._clump_pos = numpy.asarray((x0, y0, z0))
        assert clump_mass is None or isinstance(clump_mass, float)
        self._clump_mass = clump_mass
        if all(v is not None for v in (vx, vy, vz)):
            self._vel = numpy.vstack([vx, vy, vz]).T
            assert self._vel.shape == self.pos.shape
        assert m.ndim == 1 and m.size == self.Npart
        self._m = m
        assert index is None or (isinstance(index, (int, numpy.int64)) and index >= 0)  # noqa
        self._index = index
        assert rhoc is None or rhoc > 0
        self._rhoc = rhoc
        assert G is None or G > 0
        self._G = G

    @property
    def pos(self):
        """
        Cartesian particle coordinates centered at the clump.

        Returns
        -------
        pos : 2-dimensional array of shape `(n_particles, 3)`.
        """
        return self._pos

    @property
    def Npart(self):
        """
        Number of particles associated with this clump.

        Returns
        -------
        Npart : int
        """
        return self.pos.shape[0]

    @property
    def r(self):
        """
        Radial distance of the particles from the clump peak.

        Returns
        -------
        r : 1-dimensional array of shape `(n_particles, )`.
        """
        return numpy.sum(self.pos**2, axis=1)**0.5

    @property
    def rmin(self):
        """
        The minimum radial distance of a particle.

        Returns
        -------
        rmin : float
        """
        return numpy.min(self.r)

    @property
    def rmax(self):
        """
        The maximum radial distance of a particle.

        Returns
        -------
        rmin : float
        """
        return numpy.max(self.r)

    @property
    def clump_pos(self):
        """
        Cartesian position components of the clump.

        Returns
        -------
        pos : 1-dimensional array of shape `(3, )`
        """
        return self._clump_pos

    @property
    def clump_mass(self):
        """
        Clump mass.

        Returns
        -------
        mass : float
        """
        if self._clump_mass is None:
            raise ValueError("Clump mass `clump_mass` has not been set.")
        return self._clump_mass

    @property
    def vel(self):
        """
        Cartesian velocity components of the clump.

        Returns
        -------
        vel : 2-dimensional array of shape (`n_particles, 3`)
        """
        if self._vel is None:
            raise ValueError("Velocities `vel` have not been set.")
        return self._vel

    @property
    def m(self):
        """
        Particle masses.

        Returns
        -------
        m : 1-dimensional array of shape `(n_particles, )`
        """
        return self._m

    @property
    def center_mass(self):
        """
        Cartesian position components of the clump centre of mass. Note that
        this is already in a frame centered at the clump's potential minimum.

        Returns
        -------
        cm : 1-dimensional array of shape `(3, )`
        """
        return numpy.average(self.pos, axis=0, weights=self.m)

    @property
    def angular_momentum(self):
        """
        Clump angular momentum in the box coordinates.

        Returns
        -------
        J : 1-dimensional array or shape `(3, )`
        """
        J = numpy.cross(self.pos - self.center_mass, self.vel)
        return numpy.einsum("i,ij->j", self.m, J)

    @property
    def lambda200c(self):
        r"""
        Clump Bullock spin, see Eq. 5 in [1], in a radius of
        :math:`R_{\rm 200c}`.

        References
        ----------
        [1] A Universal Angular Momentum Profile for Galactic Halos; 2001;
        Bullock, J. S.;  Dekel, A.;  Kolatt, T. S.;  Kravtsov, A. V.;
        Klypin, A. A.;  Porciani, C.;  Primack, J. R.

        Returns
        -------
        lambda200c : float
        """
        J = self.angular_momentum
        R, M = self.spherical_overdensity_mass(200)
        V = numpy.sqrt(self.G * M / R)
        return numpy.linalg.norm(J) / (numpy.sqrt(2) * M * V * R)

    @property
    def index(self):
        """
        Halo finder clump index.

        Returns
        -------
        hindex : int
        """
        if self._index is None:
            raise ValueError("Halo index `hindex` has not been set.")
        return self._index

    @property
    def rhoc(self):
        r"""
        Critical density :math:`\rho_c` at this snapshot in box units.

        Returns
        -------
        rhoc : float
        """
        if self._rhoc is None:
            raise ValueError("The critical density `rhoc` has not been set.")
        return self._rhoc

    @property
    def G(self):
        r"""
        Gravitational constant :math:`G` in box units.

        Returns
        -------
        G : float
        """
        if self._G is None:
            raise ValueError("The grav. constant `G` has not been set.")
        return self._G

    @property
    def total_particle_mass(self):
        """
        Total mass of all particles.

        Returns
        -------
        tot_mass : float
        """
        return numpy.sum(self.m)

    @property
    def mean_particle_pos(self):
        """
        Mean Cartesian particle coordinate. Not centered at the halo!

        Returns
        -------
        pos : 1-dimensional array of shape `(3, )`
        """
        return numpy.mean(self.pos + self.clump_pos, axis=0)

    def enclosed_spherical_mass(self, rmax, rmin=0):
        """
        Enclosed spherical mass between two radii in box units.

        Parameters
        ----------
        rmax : float
            The maximum radial distance.
        rmin : float, optional
            The minimum radial distance. By default 0.

        Returns
        -------
        M_enclosed : float
            The enclosed mass.
        """
        return numpy.sum(self.m[(self.r >= rmin) & (self.r <= rmax)])

    def enclosed_spherical_volume(self, rmax, rmin=0):
        """
        Enclosed spherical volume within two radii in box units.

        Parameters
        ----------
        rmax : float
            Maximum radial distance.
        rmin : float, optional
            Minimum radial distance. By default 0.

        Returns
        -------
        vol : float
        """
        return 4 * numpy.pi / 3 * (rmax**3 - rmin**3)

    def spherical_overdensity_mass(self, delta, n_particles_min=10):
        r"""
        Spherical overdensity mass and radius. The mass is defined as the
        enclosed mass within a radius of where the mean enclosed spherical
        density reaches a multiple of the critical radius at a given redshift
        `self.rho_c`.

        Starts from the furthest particle, working its way inside the halo
        through an ordered list of particles. The corresponding values is the
        radial distance of the first particle whose addition sufficiently
        increases the mean density.


        Parameters
        ----------
        delta : list of int or float
            The :math:`\delta_{\rm x}` parameters where :math:`\mathrm{x}` is
            the overdensity multiple.
        n_particles_min : int
            Minimum number of enclosed particles for a radius to be
            considered trustworthy.

        Returns
        -------
        rx : float
            Radius where the enclosed density reaches required value.
        mx :  float
            Corresponding spherical enclosed mass.
        """
        # If single `delta` turn to list
        delta = [delta] if isinstance(delta, (float, int)) else delta
        # If given a list or tuple turn to array
        _istlist = isinstance(delta, (list, tuple))
        delta = numpy.asarray(delta, dtype=float) if _istlist else delta

        # Ordering of deltas
        order_delta = numpy.argsort(delta)
        # Sort the particles
        order_particles = numpy.argsort(self.r)[::-1]
        # Density to aim for
        n_delta = delta.size
        target_density = delta * self.rhoc

        # The sum of particle masses, starting from the outside
        # Adds the furtherst particle ensure that the 0th index is tot mass
        cummass_ordered = (self.total_particle_mass
                           + self.m[order_particles][0]
                           - numpy.cumsum(self.m[order_particles]))
        # Enclosed volumes at particle radii
        volumes = self.enclosed_spherical_volume(self.r[order_particles])
        densities = cummass_ordered / volumes

        # Pre-allocate arrays
        rfound = numpy.full_like(delta, numpy.nan)
        mfound = numpy.full_like(rfound, numpy.nan)

        for n in order_delta:
            overdense_mask = densities > target_density[n]

            # Enforce that we have at least several particles enclosed
            if numpy.sum(overdense_mask) < n_particles_min:
                continue
            # The outermost particle radius where the overdensity is achieved
            k = numpy.where(overdense_mask)[0][0]
            rfound[n] = self.r[order_particles][k]
            mfound[n] = cummass_ordered[k]

        # If only one delta return simply numbers
        if n_delta == 1:
            rfound = rfound[0]
            mfound = mfound[0]

        return rfound, mfound

    @classmethod
    def from_arrays(cls, particles, clump, rhoc=None, G=None):
        r"""
        Initialises `Clump` from `particles` containing the relevant particle
        information and its `clump` information.

        Paramaters
        ----------
        particles : structured array
            Array of particles belonging to this clump. Must contain
            `["x", "y", "z", "M"]` and optionally also `["vx", "vy", "vz"]`.
        clump : array
            A slice of a `clumps` array corresponding to this clump. Must
            contain `["peak_x", "peak_y", "peak_z", "mass_cl"]`.
        rhoc : float, optional
            The critical density :math:`\rho_c` at this snapshot in box units.
            By default not set.
        G : float, optional
            The gravitational constant :math:`G` in box units. By default not
            set.

        Returns
        -------
        clump : `Clump`
        """
        x, y, z, m = (particles[p] for p in ["x", "y", "z", "M"])
        x0, y0, z0, cl_mass, hindex = (
            clump[p] for p in ["peak_x", "peak_y", "peak_z", "mass_cl",
                               "index"])
        try:
            vx, vy, vz = (particles[p] for p in ["vx", "vy", "vz"])
        except ValueError:
            vx, vy, vz = None, None, None
        return cls(x, y, z, m, x0, y0, z0, cl_mass,
                   vx, vy, vz, hindex, rhoc, G)
