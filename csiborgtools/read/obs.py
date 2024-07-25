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
Scripts to read in observation.
"""
from abc import ABC, abstractmethod
from os.path import join
from warnings import warn

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy import constants

from ..utils import fprint, radec_to_cartesian

###############################################################################
#                           Text survey base class                            #
###############################################################################


class TextSurvey:
    """
    Base survey class for extracting data from text files.
    """
    _data = None
    _cosmo = None

    @property
    def data(self):
        """
        Cluster catalogue.

        Returns
        -------
        cat : structured array
            Catalogue.
        """
        if self._data is None:
            raise ValueError("`data` is not set!")
        return self._data

    @property
    def cosmo(self):
        """Desired cosmology."""
        if self._cosmo is None:
            raise ValueError("`cosmo` is not set!")
        return self._cosmo

    @property
    def keys(self):
        """Catalogue keys."""
        return self.data.dtype.names

    def __getitem__(self, key):
        return self._data[key]


###############################################################################
#                              2M++ galaxies                                  #
###############################################################################


class TwoMPPGalaxies(TextSurvey):
    """
    The 2M++ galaxy redshift catalogue [1], with the catalogue at [2].
    Removes fake galaxies used to fill the zone of avoidance. Note that the
    stated redshift is in the CMB frame.

    Parameters
    ----------
    fpath : str, optional.
        File path to the catalogue. By default
        `/mnt/extraspace/rstiskalek/catalogs/2M++_galaxy_catalog.dat`.

    References
    ----------
    [1] The 2M++ galaxy redshift catalogue; Lavaux, Guilhem, Hudson, Michael J.
    [2] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/416/2840#/article
    [3] Improving NASA/IPAC Extragalactic Database Redshift Calculations
        (2021); Anthony Carr and Tamara Davis
    """
    name = "2M++_galaxies"

    def __init__(self, fpath=None):
        if fpath is None:
            fpath = join("/mnt/extraspace/rstiskalek/catalogs/"
                         "2M++_galaxy_catalog.dat")
        self._set_data(fpath)

    def _set_data(self, fpath):
        from scipy.constants import c

        # Read the catalogue and select non-fake galaxies
        cat = np.genfromtxt(fpath, delimiter="|", )
        cat = cat[cat[:, 12] == 0, :]
        # Pre=allocate array and fillt it
        cols = [("RA", np.float64), ("DEC", np.float64),
                ("Ksmag", np.float64), ("ZCMB", np.float64)]
        data = cols_to_structured(cat.shape[0], cols)
        data["RA"] = cat[:, 1]
        data["DEC"] = cat[:, 2]
        data["Ksmag"] = cat[:, 5]
        data["ZCMB"] = cat[:, 7] / (c * 1e-3)
        self._data = data


###############################################################################
#                             2M++ groups                                     #
###############################################################################


class TwoMPPGroups(TextSurvey):
    """
    The 2M++ galaxy group catalogue [1], with the catalogue at [2].

    Parameters
    ----------
    fpath : str, optional
        File path to the catalogue. By default
        `/mnt/extraspace/rstiskalek/catalogs/2M++_group_catalog.dat`

    References
    ----------
    [1] The 2M++ galaxy redshift catalogue; Lavaux, Guilhem, Hudson, Michael J.
    [2] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/416/2840#/article
    [3] Improving NASA/IPAC Extragalactic Database Redshift Calculations
        (2021); Anthony Carr and Tamara Davis
    """
    name = "2M++_groups"

    def __init__(self, fpath=None):
        if fpath is None:
            fpath = join("/mnt/extraspace/rstiskalek/catalogs",
                         "2M++_group_catalog.dat")
        self._set_data(fpath)

    def _set_data(self, fpath):
        cat = np.genfromtxt(fpath, delimiter="|", )
        # Pre-allocate and fill the array
        cols = [("RA", np.float64), ("DEC", np.float64),
                ("K2mag", np.float64), ("Rich", np.int64),
                ("sigma", np.float64)]
        data = cols_to_structured(cat.shape[0], cols)
        data["K2mag"] = cat[:, 3]
        data["Rich"] = cat[:, 4]
        data["sigma"] = cat[:, 7]

        # Convert galactic coordinates to RA, dec
        glon = cat[:, 1]
        glat = cat[:, 2]
        coords = SkyCoord(l=glon * units.degree, b=glat * units.degree,
                          frame='galactic')
        coords = coords.transform_to("icrs")
        data["RA"] = coords.ra
        data["DEC"] = coords.dec
        self._data = data


###############################################################################
#                             FITS base class                                 #
###############################################################################


class FitsSurvey(ABC):
    """
    Base class for extracting data from FITS files. Contains two sets of
    keys: `routine_keys` and `fits_keys`. The former are user-defined
    properties calculated from the FITS file data. Both are accesible via
    `self[key]`.
    """
    _file = None
    _h = None
    _routines = None
    _selection_mask = None

    @property
    def file(self):
        """The survey FITS file."""
        if self._file is None:
            raise ValueError("`file` is not set!")
        return self._file

    @property
    def h(self):
        """Little h."""
        return self._h

    @h.setter
    def h(self, h):
        self._h = h

    @staticmethod
    def _check_in_list(member, members, kind):
        """
        Check that `member` is a member of a list `members`, `kind` is a
        member type name.
        """
        if member not in members:
            raise ValueError("Unknown {} `{}`, must be one of `{}`."
                             .format(kind, member, members))

    @property
    def routines(self):
        """
        Processing routines.

        Returns
        -------
        routines : dict
            Dictionary of routines. Keys are functions and values are their
            arguments.
        """
        return self._routines

    @property
    @abstractmethod
    def size(self):
        """Number of samples in the catalogue."""
        pass

    @property
    def masked_size(self):
        if self.selection_mask is None:
            return self.size
        return np.sum(self.selection_mask)

    @property
    def selection_mask(self):
        """Selection mask, generated with `fmask` when initialised."""
        return self._selection_mask

    @selection_mask.setter
    def selection_mask(self, mask):
        if not (isinstance(mask, np.ndarray)
                and mask.ndim == 1
                and mask.dtype == bool):
            raise TypeError("`selection_mask` must be a 1-dimensional boolean "
                            "array. Check output of `fmask`.")
        self._selection_mask = mask

    @property
    def fits_keys(self):
        """Keys of the FITS file `self.file`."""
        return self.file[1].data.columns.names

    @property
    def routine_keys(self):
        """Routine keys."""
        return list(self.routines.keys())

    def get_fitsitem(self, key):
        """Get a column `key` from the FITS file `self.file`."""
        return self.file[1].data[key]

    @property
    def keys(self):
        """Routine and FITS keys."""
        return self.routine_keys + self.fits_keys + ["INDEX"]

    def make_mask(self, steps):
        """
        Make a survey mask from a series of steps, expected to look as below.

        ```
        def steps(cls):
            return [(lambda x: cls[x], ("IN_DR7_LSS",)),
                    (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                   ]
        ```

        Parameters
        ----------
        steps : list of steps
            Selection steps.

        Returns
        -------
        mask : 1-dimensional boolean array
        """
        out = None
        steps = steps(self)
        for i, step in enumerate(steps):
            func, args = step
            if i == 0:
                out = func(*args)
            else:
                out = out & func(*args)
        return out

    def __getitem__(self, key):
        if key == "INDEX":
            mask = self.selection_mask
            if mask is None:
                return np.arange(self.size)
            else:
                return np.arange(mask.size)[mask]

        # Check duplicates
        if key in self.routine_keys and key in self.fits_keys:
            warn(f"Key `{key}` found in both `routine_keys` and `fits_keys`. "
                 "Returning `routine_keys` value.")

        if key in self.routine_keys:
            func, args = self.routines[key]
            out = func(*args)
        elif key in self.fits_keys:
            warn(f"Returning a FITS property `{key}`. "
                 "Be careful about little h!")
            out = self.get_fitsitem(key)
        else:
            raise KeyError(f"Unrecognised key `{key}`.")

        if self.selection_mask is None:
            return out
        return out[self.selection_mask]

    def __len__(self):
        return self.size


###############################################################################
#                            Planck clusters                                  #
###############################################################################


class PlanckClusters(FitsSurvey):
    r"""
    Planck 2nd Sunyaev-Zeldovich source catalogue [1].

    Parameters
    ----------
    fpath : str, optional
        Path to the FITS file. By default
        `/mnt/extraspace/rstiskalek/catalogs/HFI_PCCS_SZ-union_R2.08.fits`.
    h : float, optional
        Little h. By default `h = 0.7`. The catalogue assumes this value.
        The routine properties should take care of little h conversion.
    sel_steps : py:function:
        Steps to mask the survey. Expected to look for example like
        ```
        def steps(cls):
            return [(lambda x: cls[x], ("IN_DR7_LSS",)),
                    (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                   ]
        ```

    References
    ----------
    [1] https://heasarc.gsfc.nasa.gov/W3Browse/all/plancksz2.html
    """
    name = "Planck_clusters"
    _hdata = 0.7  # little h value of the data

    def __init__(self, fpath=None, h=0.7, sel_steps=None):
        if fpath is None:
            fpath = join("/mnt/extraspace/rstiskalek/catalogs/",
                         "HFI_PCCS_SZ-union_R2.08.fits")
        self._file = fits.open(fpath, memmap=False)
        self.h = h

        self._routines = {}
        # Set MSZ routines
        for key in ("MSZ", "MSZ_ERR_UP", "MSZ_ERR_LOW"):
            self._routines.update({key: (self._mass, (key,))})

        # Add masking. Do this at the end!
        if sel_steps is not None:
            self.selection_mask = self.make_mask(sel_steps)

    @property
    def size(self):
        return self.get_fitsitem("MSZ").size

    def _mass(self, key):
        """Get mass. Puts in units of 1e14 and converts little h."""
        return self.get_fitsitem(key) * 1e14 * (self._hdata / self.h)**2

    def match_to_mcxc(self, mcxc):
        """
        Return the MCXC catalogue indices of the Planck catalogue detections.
        Finds the index of the quoted Planck MCXC counterpart in the MCXC
        array. If not found throws an error. For this reason it may be better
        to make sure the MCXC catalogue reaches further.

        Parameters
        ----------
        mcxc : :py:class`MCXCClusters`
            MCXC cluster object.

        Returns
        -------
        indxs : list of int
            Array of MCXC indices to match the Planck array. If no counterpart
            is found returns `np.nan`.
        """
        if not isinstance(mcxc, MCXCClusters):
            raise TypeError("`mcxc` must be `MCXCClusters` type.")

        # Planck MCXC need to be decoded to str
        planck_names = [name.decode() for name in self["MCXC"]]
        mcxc_names = [name for name in mcxc["MCXC"]]

        indxs = [np.nan] * len(planck_names)
        for i, name in enumerate(planck_names):
            if name == "":
                continue
            if name in mcxc_names:
                indxs[i] = mcxc_names.index(name)
            else:
                raise ValueError("Planck MCXC identifier `{}` not found in "
                                 "the MCXC catalogue.".format(name))
        return indxs


###############################################################################
#                             MCXC Clusters                                   #
###############################################################################


class MCXCClusters(FitsSurvey):
    r"""
    MCXC Meta-Catalog of X-Ray Detected Clusters of Galaxies catalogue [1],
    with data description at [2] and download at [3].

    Note
    ----
    The exact mass conversion has non-trivial dependence on :math:`H(z)`, see
    [1] for more details. However, this should be negligible.

    Parameters
    ----------
    fpath : str, optional
        Path to the source catalogue obtained from [3]. Expected to be the fits
        file. By default `/mnt/extraspace/rstiskalek/catalogs/mcxc.fits`.
    h : float, optional
        Little h. By default `h = 0.7`. The catalogue assumes this value.
        The routine properties should take care of little h conversion.
    sel_steps : py:function:
        Steps to mask the survey. Expected to look for example like
        ```
            steps = [(lambda x: cls[x], ("IN_DR7_LSS",)),
                     (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                     ]
        ```.


    References
    ----------
    [1] The MCXC: a meta-catalogue of x-ray detected clusters of galaxies
        (2011); Piffaretti, R. ;  Arnaud, M. ;  Pratt, G. W. ;  Pointecouteau,
        E. ;  Melin, J. -B.
    [2] https://heasarc.gsfc.nasa.gov/W3Browse/rosat/mcxc.html
    [3] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/534/A109#/article
    """
    name = "MCXC"
    _hdata = 0.7  # Little h of the catalogue

    def __init__(self, fpath=None, h=0.7, sel_steps=None):
        if fpath is None:
            fpath = "/mnt/extraspace/rstiskalek/catalogs/mcxc.fits"
        self._file = fits.open(fpath, memmap=False)
        self.h = h
        # Set mass and luminosity routines
        self._routines = {}
        self._routines.update({"M500": (self._mass, ("M500",))})
        self._routines.update({"L500": (self._lum, ("L500",))})

        if sel_steps is not None:
            self.selection_mask = self.make_mask(sel_steps)

    @property
    def size(self):
        return self.get_fitsitem("M500").size

    def _mass(self, key):
        """Get mass. Put in units of 1e14 Msun back and convert little h."""
        return self.get_fitsitem(key) * 1e14 * (self._hdata / self.h)**2

    def _lum(self, key):
        """Get luminosity, puts back units to be in ergs/s."""
        return self.get_fitsitem(key) * 1e44 * (self._hdata / self.h)**2


###############################################################################
#                              SDSS galaxies                                  #
###############################################################################


class SDSS(FitsSurvey):
    """
    SDSS data manipulations. Data obtained from [1]. Carries routines for
    ABSMAG, APPMAG, COL, DIST, MTOL.

    Parameters
    ----------
    fpath : str, optional
        Path to the FITS file.
    h : float, optional
        Little h. By default `h = 1`. The catalogue assumes this value.
        The routine properties should take care of little h conversion.
    Om0 : float, optional
        Matter density. By default `Om0 = 0.3175`, matching CSiBORG.
    sel_steps : py:function:
        Steps to mask the survey. Expected to look for example like
        ```
            steps = [(lambda x: cls[x], ("IN_DR7_LSS",)),
                     (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                     ]
        ```.

    References
    ----------
    [1] https://www.sdss.org/dr13/manga/manga-target-selection/nsa/
    """
    name = "SDSS"

    def __init__(self, fpath, h=1, Om0=0.3175, sel_steps=None):
        self._file = fits.open(fpath, memmap=False)
        self.h = h

        # Survey bands and photometries
        self._bands = ['F', 'N', 'u', 'g', 'r', 'i', 'z']
        self._photos = ["SERSIC", "ELPETRO"]

        self._routines = {}
        # Set ABSMAGroutines
        for photo in self._photos:
            for band in self._bands:
                # ABSMAG
                key = "{}_ABSMAG_{}".format(photo, band)
                val = (self._absmag, (photo, band))
                self.routines.update({key: val})
        # Set APPMAG routines
        for photo in self._photos:
            for band in self._bands:
                key = "{}_APPMAG_{}".format(photo, band)
                val = (self._appmag, (photo, band))
                self.routines.update({key: val})
        # Set COL routines
        for photo in self._photos:
            for band1 in self._bands:
                for band2 in self._bands:
                    key = "{}_COL_{}{}".format(photo, band1, band2)
                    val = (self._colour, (photo, band1, band2))
                    self.routines.update({key: val})
        # Set DIST routine
        self.routines.update({"DIST": (self._dist, ())})
        self.routines.update(
            {"DIST_UNCORRECTED": (self._dist_uncorrected, (Om0,))})
        # Set MASS routines
        for photo in self._photos:
            key = "{}_MASS".format(photo)
            val = (self._solmass, (photo,))
            self.routines.update({key: val})
        # Set MTOL
        for photo in self._photos:
            for band in self._bands:
                key = "{}_MTOL_{}".format(photo, band)
                val = (self._mtol, (photo, band))
                self.routines.update({key: val})
        # Set IN_DR7_LSS
        self.routines.update({"IN_DR7_LSS": (self._in_dr7_lss, ())})

        # Add masking. Do this at the end!
        if sel_steps is not None:
            self.selection_mask = self.make_mask(sel_steps)

    @property
    def size(self):
        mask = self.selection_mask
        if mask is not None:
            return np.sum(mask)
        else:
            return self.get_fitsitem("ZDIST").size

    def _absmag(self, photo, band):
        """
        Get absolute magnitude of a given photometry and band. Converts to
        the right little h.
        """
        self._check_in_list(photo, self._photos, "photometry")
        self._check_in_list(band, self._bands, "band")
        k = self._bands.index(band)
        mag = self.get_fitsitem("{}_ABSMAG".format(photo))[:, k]
        return mag + 5 * np.log10(self.h)

    def _kcorr(self, photo, band):
        """
        Get K-correction of a given photometry and band.
        """
        self._check_in_list(photo, self._photos, "photometry")
        self._check_in_list(band, self._bands, "band")
        k = self._bands.index(band)
        return self.get_fitsitem("{}_KCORRECT".format(photo))[:, k]

    def _appmag(self, photo, band):
        """
        Get apparent magnitude of a given photometry and band.
        """
        lumdist = (1 + self.get_fitsitem("ZDIST")) * self._dist()
        absmag = self._absmag(photo, band)
        kcorr = self._kcorr(photo, band)
        return absmag + 25 + 5 * np.log10(lumdist) + kcorr

    def _colour(self, photo, band1, band2):
        """
        Get colour of a given photometry, i.e. `band1` - `band2` absolute
        magnitude.
        """
        return self._absmag(photo, band1) - self._absmag(photo, band2)

    def _dist(self):
        r"""
        Get the corresponding distance estimate from `ZDIST`, defined as below.

            "Distance estimate using pecular velocity model of Willick et al.
            (1997), expressed as a redshift equivalent; multiply by c/H0 for
            Mpc"

        Distance is converted to math:`h != 1` units.
        """
        return self.get_fitsitem("ZDIST") * constants.c * 1e-3 / (100 * self.h)

    def _dist_uncorrected(self, Om0):
        """
        Get the comoving distance estimate from `Z`, i.e. redshift uncorrected
        for peculiar motion in the heliocentric frame.
        """
        cosmo = FlatLambdaCDM(H0=100 * self.h, Om0=Om0)
        return cosmo.comoving_distance(self.get_fitsitem("Z")).value

    def _solmass(self, photo):
        """
        Get solar mass of a given photometry. Converts little h.
        """
        self._check_in_list(photo, self._photos, "photometry")
        return self.get_fitsitem("{}_MASS".format(photo)) / self.h**2

    def _mtol(self, photo, band):
        """
        Get mass-to-light ratio of a given photometry. Converts little h.
        """
        self._check_in_list(photo, self._photos, "photometry")
        self._check_in_list(band, self._bands, "band")
        k = self._bands.index(band)
        return self.get_fitsitem("{}_MTOL".format(photo))[:, k] / self.h**2

    def _in_dr7_lss(self):
        """
        Get `IN_DR7_LSS` and turn to a boolean array.
        """
        return self.get_fitsitem("IN_DR7_LSS").astype(bool)


###############################################################################
#                           Individual observations                           #
###############################################################################


class BaseSingleObservation(ABC):
    """
    Base class to hold information about a single object.
    """
    def __init__(self):
        self._spherical_pos = None
        self._name = None

    @property
    def spherical_pos(self):
        """
        Spherical position of the observation in dist/RA/dec in Mpc / h and
        degrees, respectively.

        Returns
        -------
        1-dimensional array of shape (3,)
        """
        if self._spherical_pos is None:
            raise ValueError("`spherical_pos` is not set!")
        return self._spherical_pos

    @spherical_pos.setter
    def spherical_pos(self, pos):
        if isinstance(pos, (list, tuple)):
            pos = np.array(pos)

        if not pos.shape == (3,):
            raise ValueError("`spherical_pos` must be a of shape (3,).")

        self._spherical_pos = pos

    @property
    def mass(self):
        """Total mass estimate in Msun / h."""
        if self._mass is None:
            raise ValueError("`mass` is not set!")
        return self._mass

    @mass.setter
    def mass(self, mass):
        if not isinstance(mass, (int, float)):
            raise ValueError("`mass` must be a float.")
        self._mass = mass

    def cartesian_pos(self, boxsize):
        """
        Cartesian position in Mpc / h, assuming the observer is in the centre
        of the box.
        """
        return radec_to_cartesian(
            self.spherical_pos.reshape(1, 3)).reshape(-1,) + boxsize / 2

    @property
    def name(self):
        """Object name."""
        if self._name is None:
            raise ValueError("`name` is not set!")
        return self._name

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise ValueError("`name` must be a string.")
        self._name = name


class ObservedCluster(BaseSingleObservation):
    """
    Class to hold information about an observed cluster.

    Parameters
    ----------
    RA : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    dist : float
        Distance in Mpc / h.
    mass : float
        Total mass estimate in Msun / h.
    name : str
        Cluster name.
    """
    def __init__(self, RA, dec, dist, mass, name):
        super().__init__()
        self.name = name
        self.spherical_pos = [dist, RA, dec]
        self.mass = mass


###############################################################################
#                           Pantheon+ data                                    #
###############################################################################


def read_pantheonplus_covariance(fname, ww, ):
    """Read in a Pantheon+ covariance matrix."""
    origlen = len(ww)
    # Pantheon+SH0ES routine to read in the covariance matrix
    with open(fname) as f:
        # Keep this line, otherwise will fail
        line = f.readline()  # noqa
        n = int(np.sum(ww))
        C = np.zeros((n, n))
        ii = -1
        jj = -1
        for i in range(origlen):
            jj = -1
            if ww[i]:
                ii += 1
            for j in range(origlen):
                if ww[j]:
                    jj += 1
                val = float(f.readline())
                if ww[i]:
                    if ww[j]:
                        C[ii, jj] = val

    return C


def read_pantheonplus_data(fname_data, fname_covmat_statsys, fname_covmat_vpec,
                           subtract_vpec, verbose=True):
    """Read in the Pantheon+ covariance matrix."""
    fprint("reading the Pantheon+ data.", verbose)
    data = np.genfromtxt(fname_data, names=True, dtype=None, encoding=None)
    ww = np.ones(len(data), dtype=bool)

    fprint("reading the Pantheon+ STAT+SYS covariance matrix.", verbose)
    C = read_pantheonplus_covariance(fname_covmat_statsys, ww)

    if subtract_vpec:
        fprint("reading the Pantheon+ VPEC covariance matrix.", verbose)
        C_vpec = read_pantheonplus_covariance(fname_covmat_vpec, ww)

    # Subtracting the VPEC covariance matrix from the STAT+SYS covariance
    # matrix produces negative eigenvalues. Emailed Maria to ask about this.

    return data, C, C_vpec


###############################################################################
#                           Utility functions                                 #
###############################################################################

def match_array_to_no_masking(arr, surv):
    """
    Match an array to a survey without masking.

    Parameters
    ----------
    arr : n-dimensional array
        Array to match.
    surv : survey class
        Survey class.

    Returns
    -------
    out : n-dimensional array
    """
    dtype = arr.dtype
    if arr.ndim > 1:
        shape = arr.shape
        out = np.full((surv.selection_mask.size, *shape[1:]), np.nan,
                      dtype=dtype)
    else:
        out = np.full(surv.selection_mask.size, np.nan, dtype=dtype)

    for i, indx in enumerate(surv["INDEX"]):
        out[indx] = arr[i]

    return out


def cols_to_structured(N, cols):
    """
    Allocate a structured array from `cols`, a list of (name, dtype) tuples.
    """
    if not (isinstance(cols, list)
            and all(isinstance(c, tuple) and len(c) == 2 for c in cols)):
        raise TypeError("`cols` must be a list of (name, dtype) tuples.")

    names, formats = zip(*cols)
    dtype = {"names": names, "formats": formats}

    return np.full(N, np.nan, dtype=dtype)
