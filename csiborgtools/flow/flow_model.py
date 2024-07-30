# Copyright (C) 2024 Richard Stiskalek
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
Validation of the CSiBORG velocity field against PV measurements. A lot here
is based on [1], though with many modifications. Throughout, comoving distances
are in `Mpc / h` and velocities in `km / s`.

References
----------
[1] https://arxiv.org/abs/1912.09383.
"""
from abc import ABC, abstractmethod

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from h5py import File
from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import logsumexp
from numpyro import deterministic, factor, sample
from numpyro.distributions import Normal, Uniform
from quadax import simpson
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from tqdm import trange

from ..params import SPEED_OF_LIGHT, simname2Omega_m
from ..utils import fprint, radec_to_galactic, radec_to_supergalactic

H0 = 100  # km / s / Mpc


###############################################################################
#                             Data loader                                     #
###############################################################################


class DataLoader:
    """
    Data loader for the line of sight (LOS) interpolated fields and the
    corresponding catalogues.

    Parameters
    ----------
    simname : str
        Simulation name.
    ksim : int or list of int
        Index of the simulation to read in (not the IC index).
    catalogue : str
        Name of the catalogue with LOS objects.
    catalogue_fpath : str
        Path to the LOS catalogue file.
    paths : csiborgtools.read.Paths
        Paths object.
    ksmooth : int, optional
        Smoothing index.
    store_full_velocity : bool, optional
        Whether to store the full 3D velocity field. Otherwise stores only
        the radial velocity.
    verbose : bool, optional
        Verbose flag.
    """
    def __init__(self, simname, ksim, catalogue, catalogue_fpath, paths,
                 ksmooth=None, store_full_velocity=False, verbose=True):
        fprint("reading the catalogue,", verbose)
        self._cat = self._read_catalogue(catalogue, catalogue_fpath)
        self._catname = catalogue

        fprint("reading the interpolated field.", verbose)
        self._field_rdist, self._los_density, self._los_velocity, self._rmax = self._read_field(  # noqa
            simname, ksim, catalogue, ksmooth, paths)

        if len(self._cat) != self._los_density.shape[1]:
            raise ValueError("The number of objects in the catalogue does not "
                             "match the number of objects in the field.")

        fprint("calculating the radial velocity.", verbose)
        nobject = self._los_density.shape[1]
        dtype = self._los_density.dtype

        if simname in ["Carrick2015", "Lilow2024"]:
            # Carrick+2015 and Lilow+2024 are in galactic coordinates
            d1, d2 = radec_to_galactic(self._cat["RA"], self._cat["DEC"])
        elif "CF4" in simname:
            # CF4 is in supergalactic coordinates
            d1, d2 = radec_to_supergalactic(self._cat["RA"], self._cat["DEC"])
        else:
            d1, d2 = self._cat["RA"], self._cat["DEC"]

        num_sims = len(self._los_density)
        radvel = np.empty((num_sims, nobject, len(self._field_rdist)), dtype)
        for k in range(num_sims):
            for i in range(nobject):
                radvel[k, i, :] = radial_velocity_los(
                    self._los_velocity[k, :, i, ...], d1[i], d2[i])
        self._los_radial_velocity = radvel

        if not store_full_velocity:
            self._los_velocity = None

        self._Omega_m = simname2Omega_m(simname)

        # Normalize the CSiBORG density by the mean matter density
        if "csiborg" in simname:
            cosmo = FlatLambdaCDM(H0=H0, Om0=self._Omega_m)
            mean_rho_matter = cosmo.critical_density0.to("Msun/kpc^3").value
            mean_rho_matter *= self._Omega_m
            self._los_density /= mean_rho_matter

        # Since Carrick+2015 and CF4 provide `rho / <rho> - 1`
        if simname in ["Carrick2015", "CF4", "CF4gp"]:
            self._los_density += 1

        # But some CF4 delta values are < -1. Check that CF4 really reports
        # this.
        if simname in ["CF4", "CF4gp"]:
            self._los_density = np.clip(self._los_density, 1e-2, None,)

        # Lilow+2024 outside of the range data is NaN. Replace it with some
        # finite values. This is OK because the PV tracers are not so far.
        if simname == "Lilow2024":
            self._los_density[np.isnan(self._los_density)] = 1.
            self._los_radial_velocity[np.isnan(self._los_radial_velocity)] = 0.

        self._mask = np.ones(len(self._cat), dtype=bool)
        self._catname = catalogue

    @property
    def cat(self):
        """The distance indicators catalogue (structured array)."""
        return self._cat[self._mask]

    @property
    def catname(self):
        """Catalogue name."""
        return self._catname

    @property
    def rdist(self):
        """Radial distances at which the field was interpolated."""
        return self._field_rdist

    @property
    def los_density(self):
        """
        Density field along the line of sight `(n_sims, n_objects, n_steps)`
        """
        return self._los_density[:, self._mask, ...]

    @property
    def los_velocity(self):
        """
        Velocity field along the line of sight `(n_sims, 3, n_objects,
        n_steps)`.
        """
        if self._los_velocity is None:
            raise ValueError("The 3D velocities were not stored.")

        return self._los_velocity[:, :, self._mask, ...]

    @property
    def rmax(self):
        """
        Radial distance above which the underlying reconstruction is
        extrapolated `(n_sims, n_objects)`.
        """
        return self._rmax[:, self._mask]

    @property
    def los_radial_velocity(self):
        """
        Radial velocity along the line of sight `(n_sims, n_objects, n_steps)`.
        """
        return self._los_radial_velocity[:, self._mask, ...]

    def _read_field(self, simname, ksims, catalogue, ksmooth, paths):
        nsims = paths.get_ics(simname)
        if isinstance(ksims, int):
            ksims = [ksims]

        if not all(0 <= ksim < len(nsims) for ksim in ksims):
            raise ValueError(f"Invalid simulation index: `{ksims}`")

        if "Pantheon+" in catalogue:
            fpath = paths.field_los(simname, "Pantheon+")
        else:
            fpath = paths.field_los(simname, catalogue)

        los_density = [None] * len(ksims)
        los_velocity = [None] * len(ksims)
        rmax = [None] * len(ksims)

        for n, ksim in enumerate(ksims):
            nsim = nsims[ksim]

            with File(fpath, 'r') as f:
                has_smoothed = True if f[f"density_{nsim}"].ndim > 2 else False
                if has_smoothed and (ksmooth is None or not isinstance(ksmooth, int)):  # noqa
                    raise ValueError("The output contains smoothed field but "
                                     "`ksmooth` is None. It must be provided.")

                indx = (..., ksmooth) if has_smoothed else (...)
                los_density[n] = f[f"density_{nsim}"][indx]
                los_velocity[n] = f[f"velocity_{nsim}"][indx]
                rdist = f[f"rdist_{nsim}"][...]
                rmax[n] = f[f"rmax_{nsim}"][indx]

        los_density = np.stack(los_density)
        los_velocity = np.stack(los_velocity)
        rmax = np.stack(rmax)

        return rdist, los_density, los_velocity, rmax

    def _read_catalogue(self, catalogue, catalogue_fpath):
        if catalogue == "A2":
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]
        elif catalogue in ["LOSS", "Foundation", "SFI_gals", "2MTF",
                           "Pantheon+", "SFI_gals_masked", "SFI_groups",
                           "Pantheon+_groups", "Pantheon+_groups_zSN",
                           "Pantheon+_zSN"]:
            with File(catalogue_fpath, 'r') as f:
                if "Pantheon+" in catalogue:
                    grp = f["Pantheon+"]
                else:
                    grp = f[catalogue]

                dtype = [(key, np.float32) for key in grp.keys()]
                arr = np.empty(len(grp["RA"]), dtype=dtype)
                for key in grp.keys():
                    arr[key] = grp[key][:]
        elif "CB2_" in catalogue:
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]
        elif "UPGLADE" in catalogue:
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    if key == "mask":
                        continue

                    arr[key] = f[key][:]
        else:
            raise ValueError(f"Unknown catalogue: `{catalogue}`.")

        return arr

    def make_jackknife_mask(self, i, n_splits, seed=42):
        """
        Set the internal jackknife mask to exclude the `i`-th split out of
        `n_splits`.
        """
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        n = len(self._cat)
        indxs = np.arange(n)

        gen = np.random.default_rng(seed)
        gen.shuffle(indxs)

        for j, (train_index, __) in enumerate(cv.split(np.arange(n))):
            if i == j:
                self._mask = indxs[train_index]
                return

        raise ValueError("The index `i` must be in the range of `n_splits`.")

    def reset_mask(self):
        """Reset the jackknife mask."""
        self._mask = np.ones(len(self._cat), dtype=bool)


###############################################################################
#                       Supplementary flow functions                          #
###############################################################################


def radial_velocity_los(los_velocity, ra, dec):
    """
    Calculate the radial velocity along the LOS from the 3D velocity
    along the LOS `(3, n_steps)`.
    """
    types = (float, np.float32, np.float64)
    if not isinstance(ra, types) and not isinstance(dec, types):
        raise ValueError("RA and dec must be floats.")

    if los_velocity.ndim != 2 and los_velocity.shape[0] != 3:
        raise ValueError("The shape of `los_velocity` must be (3, n_steps).")

    ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)

    vx, vy, vz = los_velocity
    return (vx * np.cos(ra_rad) * np.cos(dec_rad)
            + vy * np.sin(ra_rad) * np.cos(dec_rad)
            + vz * np.sin(dec_rad))


###############################################################################
#                           JAX Flow model                                    #
###############################################################################


def lognorm_mean_std_to_loc_scale(mu, std):
    """
    Calculate the location and scale parameters for the log-normal distribution
    from the mean and standard deviation.
    """
    loc = np.log(mu) - 0.5 * np.log(1 + (std / mu) ** 2)
    scale = np.sqrt(np.log(1 + (std / mu) ** 2))
    return loc, scale


def dist2redshift(dist, Omega_m):
    """
    Convert comoving distance to cosmological redshift if the Universe is
    flat and z << 1.
    """
    eta = 3 * Omega_m / 2
    return 1 / eta * (1 - (1 - 2 * H0 * dist / SPEED_OF_LIGHT * eta)**0.5)


def redshift2dist(z, Omega_m):
    """
    Convert cosmological redshift to comoving distance if the Universe is
    flat and z << 1.
    """
    q0 = 3 * Omega_m / 2 - 1
    return SPEED_OF_LIGHT * z / (2 * H0) * (2 - z * (1 + q0))


def gradient_redshift2dist(z, Omega_m):
    """
    Gradient of the redshift to comoving distance conversion if the Universe is
    flat and z << 1.
    """
    q0 = 3 * Omega_m / 2 - 1
    return SPEED_OF_LIGHT / H0 * (1 - z * (1 + q0))


def dist2distmodulus(dist, Omega_m):
    """Convert comoving distance to distance modulus, assuming z << 1."""
    zcosmo = dist2redshift(dist, Omega_m)
    luminosity_distance = dist * (1 + zcosmo)
    return 5 * jnp.log10(luminosity_distance) + 25


def distmodulus2dist(mu, Omega_m, ninterp=10000, zmax=0.1, mu2comoving=None,
                     return_interpolator=False):
    """
    Convert distance modulus to comoving distance. This is costly as it builds
    up the interpolator every time it is called, unless it is provided.

    Parameters
    ----------
    mu : float or 1-dimensional array
        Distance modulus.
    Omega_m : float
        Matter density parameter.
    ninterp : int, optional
        Number of points to interpolate the mapping from distance modulus to
        comoving distance.
    zmax : float, optional
        Maximum redshift for the interpolation.
    mu2comoving : callable, optional
        Interpolator from distance modulus to comoving distance. If not
        provided, it is built up every time the function is called.
    return_interpolator : bool, optional
        Whether to return the interpolator as well.

    Returns
    -------
    float (or 1-dimensional array) and callable (optional)
    """
    if mu2comoving is None:
        zrange = np.linspace(1e-15, zmax, ninterp)
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)
        mu2comoving = interp1d(
            cosmo.distmod(zrange).value, cosmo.comoving_distance(zrange).value,
            kind="cubic")

    if return_interpolator:
        return mu2comoving(mu), mu2comoving

    return mu2comoving(mu)


def distmodulus2redsfhit(mu, Omega_m, ninterp=10000, zmax=0.1, mu2z=None,
                         return_interpolator=False):
    """
    Convert distance modulus to cosmological redshift. This is costly as it
    builts up the interpolator every time it is called, unless it is provided.

    Parameters
    ----------
    mu : float or 1-dimensional array
        Distance modulus.
    Omega_m : float
        Matter density parameter.
    ninterp : int, optional
        Number of points to interpolate the mapping from distance modulus to
        comoving distance.
    zmax : float, optional
        Maximum redshift for the interpolation.
    mu2z : callable, optional
        Interpolator from distance modulus to cosmological redsfhit. If not
        provided, it is built up every time the function is called.
    return_interpolator : bool, optional
        Whether to return the interpolator as well.

    Returns
    -------
    float (or 1-dimensional array) and callable (optional)
    """
    if mu2z is None:
        zrange = np.linspace(1e-15, zmax, ninterp)
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)
        mu2z = interp1d(cosmo.distmod(zrange).value, zrange, kind="cubic")

    if return_interpolator:
        return mu2z(mu), mu2z

    return mu2z(mu)


def project_Vext(Vext_x, Vext_y, Vext_z, RA_radians, dec_radians):
    """Project the external velocity vector onto the line of sight."""
    cos_dec = jnp.cos(dec_radians)
    return (Vext_x * jnp.cos(RA_radians) * cos_dec
            + Vext_y * jnp.sin(RA_radians) * cos_dec
            + Vext_z * jnp.sin(dec_radians))


def predict_zobs(dist, beta, Vext_radial, vpec_radial, Omega_m):
    """
    Predict the observed redshift at a given comoving distance given some
    velocity field.
    """
    zcosmo = dist2redshift(dist, Omega_m)
    vrad = beta * vpec_radial + Vext_radial
    return (1 + zcosmo) * (1 + vrad / SPEED_OF_LIGHT) - 1


###############################################################################
#                          Flow validation models                             #
###############################################################################


def calculate_ptilde_wo_bias(xrange, mu, err_squared, r_squared_xrange):
    """Calculate `ptilde(r)` without imhomogeneous Malmquist bias."""
    ptilde = jnp.exp(-0.5 * (xrange - mu)**2 / err_squared)
    ptilde *= r_squared_xrange
    return ptilde


def calculate_likelihood_zobs(zobs, zobs_pred, e2_cz):
    """
    Calculate the likelihood of the observed redshift given the predicted
    redshift.
    """
    dcz = SPEED_OF_LIGHT * (zobs[:, None] - zobs_pred)
    return jnp.exp(-0.5 * dcz**2 / e2_cz) / jnp.sqrt(2 * np.pi * e2_cz)

###############################################################################
#                          Base flow validation                               #
###############################################################################


class BaseFlowValidationModel(ABC):

    def _setattr_as_jax(self, names, values):
        for name, value in zip(names, values):
            setattr(self, f"{name}", jnp.asarray(value))

    def _set_calibration_params(self, calibration_params):
        names = []
        values = []
        for key, value in calibration_params.items():
            if "e_" in key:
                key = key.replace("e_", "e2_")
                value = value**2

            names.append(key)
            values.append(value)

        self._setattr_as_jax(names, values)

    def _set_radial_spacing(self, r_xrange, Omega_m):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

        r2_xrange = r_xrange**2
        r2_xrange /= r2_xrange.mean()
        self.r_xrange = r_xrange
        self.r2_xrange = r2_xrange

        # Require `zmin` < 0 because the first radial step is likely at 0.
        z_xrange = z_at_value(
            cosmo.comoving_distance, r_xrange * u.Mpc, zmin=-0.01)
        mu_xrange = cosmo.distmod(z_xrange).value
        self.z_xrange = jnp.asarray(z_xrange)
        self.mu_xrange = jnp.asarray(mu_xrange)

        # Get the stepsize, we need it to be constant for Simpson's rule.
        dr = np.diff(r_xrange)
        if not np.all(np.isclose(dr, dr[0], atol=1e-5)):
            raise ValueError("The radial step size must be constant.")
        self.dr = dr[0]

    @property
    def ndata(self):
        """Number of PV objects in the catalogue."""
        return len(self.RA)

    @property
    def num_sims(self):
        """Number of simulations."""
        return len(self.los_density)

    @abstractmethod
    def __call__(self, **kwargs):
        pass


###############################################################################
#                          SNIa parameters sampling                           #
###############################################################################


def distmod_SN(mB, x1, c, mag_cal, alpha_cal, beta_cal):
    """Distance modulus of a SALT2 SN Ia."""
    return mB - mag_cal + alpha_cal * x1 - beta_cal * c


def e2_distmod_SN(e2_mB, e2_x1, e2_c, alpha_cal, beta_cal, e_mu_intrinsic):
    """Squared error on the distance modulus of a SALT2 SN Ia."""
    return (e2_mB + alpha_cal**2 * e2_x1 + beta_cal**2 * e2_c
            + e_mu_intrinsic**2)


def sample_SN(e_mu_min, e_mu_max, mag_cal_mean, mag_cal_std, alpha_cal_mean,
              alpha_cal_std, beta_cal_mean, beta_cal_std, alpha_min, alpha_max,
              sample_alpha, name):
    """Sample SNIe Tripp parameters."""
    e_mu = sample(f"e_mu_{name}", Uniform(e_mu_min, e_mu_max))
    mag_cal = sample(f"mag_cal_{name}", Normal(mag_cal_mean, mag_cal_std))
    alpha_cal = sample(
        f"alpha_cal_{name}", Normal(alpha_cal_mean, alpha_cal_std))
    beta_cal = sample(f"beta_cal_{name}", Normal(beta_cal_mean, beta_cal_std))
    alpha = sample(f"alpha_{name}",
                   Uniform(alpha_min, alpha_max)) if sample_alpha else 1.0

    return {"e_mu": e_mu,
            "mag_cal": mag_cal,
            "alpha_cal": alpha_cal,
            "beta_cal": beta_cal,
            "alpha": alpha
            }


###############################################################################
#                          Tully-Fisher parameters sampling                   #
###############################################################################

def distmod_TFR(mag, eta, a, b):
    """Distance modulus of a TFR calibration."""
    return mag - (a + b * eta)


def e2_distmod_TFR(e2_mag, e2_eta, b, e_mu_intrinsic):
    """Squared error on the TFR distance modulus."""
    return e2_mag + b**2 * e2_eta + e_mu_intrinsic**2


def sample_TFR(e_mu_min, e_mu_max, a_mean, a_std, b_mean, b_std, alpha_min,
               alpha_max, sample_alpha, name):
    """Sample Tully-Fisher calibration parameters."""
    e_mu = sample(f"e_mu_{name}", Uniform(e_mu_min, e_mu_max))
    a = sample(f"a_{name}", Normal(a_mean, a_std))
    b = sample(f"b_{name}", Normal(b_mean, b_std))
    alpha = sample(f"alpha_{name}",
                   Uniform(alpha_min, alpha_max)) if sample_alpha else 1.0

    return {"e_mu": e_mu,
            "a": a,
            "b": b,
            "alpha": alpha
            }

###############################################################################
#                    Calibration parameters sampling                          #
###############################################################################


def sample_calibration(Vext_min, Vext_max, Vmono_min, Vmono_max, beta_min,
                       beta_max, sigma_v_min, sigma_v_max, sample_Vmono,
                       sample_beta, sample_sigma_v_ext):
    """Sample the flow calibration."""
    Vext = sample("Vext", Uniform(Vext_min, Vext_max).expand([3]))
    sigma_v = sample("sigma_v", Uniform(sigma_v_min, sigma_v_max))

    beta = sample("beta", Uniform(beta_min, beta_max)) if sample_beta else 1.0                                 # noqa
    Vmono = sample("Vmono", Uniform(Vmono_min, Vmono_max)) if sample_Vmono else 0.0                            # noqa
    sigma_v_ext = sample("sigma_v_ext", Uniform(sigma_v_min, sigma_v_max)) if sample_sigma_v_ext else sigma_v  # noqa

    return {"Vext": Vext,
            "Vmono": Vmono,
            "sigma_v": sigma_v,
            "sigma_v_ext": sigma_v_ext,
            "beta": beta}


###############################################################################
#                            PV calibration model                             #
###############################################################################


def find_extrap_mask(rmax, rdist):
    """
    Make a mask of shape `(nsim, ngal, nrdist)` of which velocity field values
    are extrapolated. above which the
    """
    nsim, ngal = rmax.shape
    extrap_mask = np.zeros((nsim, ngal, len(rdist)), dtype=bool)
    extrap_weights = np.ones((nsim, ngal, len(rdist)))
    for i in range(nsim):
        for j in range(ngal):
            k = np.searchsorted(rdist, rmax[i, j])
            extrap_mask[i, j, k:] = True
            extrap_weights[i, j, k:] = rmax[i, j] / rdist[k:]

    return extrap_mask, extrap_weights


class PV_LogLikelihood(BaseFlowValidationModel):
    """
    Peculiar velocity validation model log-likelihood.

    Parameters
    ----------
    los_density : 3-dimensional array of shape (n_sims, n_objects, n_steps)
        LOS density field.
    los_velocity : 3-dimensional array of shape (n_sims, n_objects, n_steps)
        LOS radial velocity field.
    rmax : 1-dimensional array of shape (n_sims, n_objects)
        Radial distance above which the underlying reconstruction is
        extrapolated.
    RA, dec : 1-dimensional arrays of shape (n_objects)
        Right ascension and declination in degrees.
    z_obs : 1-dimensional array of shape (n_objects)
        Observed redshifts.
    e_zobs : 1-dimensional array of shape (n_objects)
        Errors on the observed redshifts.
    calibration_params: dict
        Calibration parameters of each object.
    r_xrange : 1-dimensional array
        Radial distances where the field was interpolated for each object.
    Omega_m : float
        Matter density parameter.
    """

    def __init__(self, los_density, los_velocity, rmax, RA, dec, z_obs,
                 e_zobs, calibration_params, r_xrange, Omega_m, kind, name):
        if e_zobs is not None:
            e2_cz_obs = jnp.asarray((SPEED_OF_LIGHT * e_zobs)**2)
        else:
            e2_cz_obs = jnp.zeros_like(z_obs)

        # Convert RA/dec to radians.
        RA = np.deg2rad(RA)
        dec = np.deg2rad(dec)

        names = ["los_density", "los_velocity", "rmax", "RA", "dec", "z_obs",
                 "e2_cz_obs"]
        values = [los_density, los_velocity, rmax, RA, dec, z_obs, e2_cz_obs]
        self._setattr_as_jax(names, values)
        self._set_calibration_params(calibration_params)
        self._set_radial_spacing(r_xrange, Omega_m)

        self.kind = kind
        self.name = name
        self.Omega_m = Omega_m
        self.norm = - self.ndata * jnp.log(self.num_sims)

        extrap_mask, extrap_weights = find_extrap_mask(rmax, r_xrange)
        self.extrap_mask = jnp.asarray(extrap_mask)
        self.extrap_weights = jnp.asarray(extrap_weights)

    def __call__(self, field_calibration_params, distmod_params,
                 sample_sigma_v_ext):
        """PV validation model log-likelihood."""
        # Turn e2_cz to be of shape (nsims, ndata, nxrange) and apply
        # sigma_v_ext where applicable
        sigma_v = field_calibration_params["sigma_v"]
        sigma_v_ext = field_calibration_params["sigma_v_ext"]
        e2_cz = jnp.full_like(self.extrap_mask, sigma_v**2, dtype=jnp.float32)
        if sample_sigma_v_ext:
            e2_cz = e2_cz.at[self.extrap_mask].set(sigma_v_ext**2)

        # Now add the observational errors
        e2_cz += self.e2_cz_obs[None, :, None]

        Vext = field_calibration_params["Vext"]
        Vmono = field_calibration_params["Vmono"]
        Vext_rad = project_Vext(Vext[0], Vext[1], Vext[2], self.RA, self.dec)

        e_mu = distmod_params["e_mu"]
        alpha = distmod_params["alpha"]
        if self.kind == "SN":
            mag_cal = distmod_params["mag_cal"]
            alpha_cal = distmod_params["alpha_cal"]
            beta_cal = distmod_params["beta_cal"]
            mu = distmod_SN(
                self.mB, self.x1, self.c, mag_cal, alpha_cal, beta_cal)
            squared_e_mu = e2_distmod_SN(
                self.e2_mB, self.e2_x1, self.e2_c, alpha_cal, beta_cal, e_mu)
        elif self.kind == "TFR":
            a = distmod_params["a"]
            b = distmod_params["b"]
            mu = distmod_TFR(self.mag, self.eta, a, b)
            squared_e_mu = e2_distmod_TFR(self.e2_mag, self.e2_eta, b, e_mu)
        else:
            raise ValueError(f"Unknown kind: `{self.kind}`.")

        # Calculate p(r) (Malmquist bias). Shape is (ndata, nxrange)
        ptilde = jnp.transpose(vmap(calculate_ptilde_wo_bias, in_axes=(0, None, None, 0))(self.mu_xrange, mu, squared_e_mu, self.r2_xrange))  # noqa
        # Inhomogeneous Malmquist bias. Shape is (n_sims, ndata, nxrange)
        ptilde = self.los_density**alpha * ptilde
        # Normalization of p(r). Shape is (n_sims, ndata)
        pnorm = simpson(ptilde, dx=self.dr, axis=-1)

        # Calculate z_obs at each distance. Shape is (n_sims, ndata, nxrange)
        # The weights are related to the extrapolation of the velocity field.
        vrad = field_calibration_params["beta"] * self.los_velocity
        vrad += (Vext_rad[None, :, None] + Vmono) * self.extrap_weights
        zobs = (1 + self.z_xrange[None, None, :]) * (1 + vrad / SPEED_OF_LIGHT) - 1  # noqa

        ptilde *= calculate_likelihood_zobs(self.z_obs, zobs, e2_cz)
        ll = jnp.log(simpson(ptilde, dx=self.dr, axis=-1)) - jnp.log(pnorm)

        return jnp.sum(logsumexp(ll, axis=0)) + self.norm


def PV_validation_model(models, distmod_hyperparams_per_model,
                        field_calibration_hyperparams):
    """
    Peculiar velocity validation NumPyro model.

    Parameters
    ----------
    models : list of `PV_LogLikelihood`
        List of PV validation log-likelihoods for each catalogue.
    distmod_hyperparams_per_model : list of dict
        Distance modulus hyperparameters for each model/catalogue.
    field_calibration_hyperparams : dict
        Field calibration hyperparameters.
    """
    field_calibration_params = sample_calibration(
        **field_calibration_hyperparams)
    sample_sigma_v_ext = field_calibration_hyperparams["sample_sigma_v_ext"]

    ll = 0.0
    for n in range(len(models)):
        model = models[n]
        distmod_hyperparams = distmod_hyperparams_per_model[n]

        if model.kind == "TFR":
            distmod_params = sample_TFR(**distmod_hyperparams, name=model.name)
        elif model.kind == "SN":
            distmod_params = sample_SN(**distmod_hyperparams, name=model.name)
        else:
            raise ValueError(f"Unknown kind: `{model.kind}`.")

        ll += model(
            field_calibration_params, distmod_params, sample_sigma_v_ext)

    deterministic("ll_values", ll)
    factor("ll", ll)


###############################################################################
#                       Shortcut to create a model                           #
###############################################################################


def get_model(loader, zcmb_max=None, verbose=True):
    """
    Get a model and extract the relevant data from the loader.

    Parameters
    ----------
    loader : DataLoader
        DataLoader instance.
    zcmb_max : float, optional
        Maximum observed redshift in the CMB frame to include.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    model : NumPyro model
    """
    zcmb_max = np.infty if zcmb_max is None else zcmb_max

    los_overdensity = loader.los_density
    los_velocity = loader.los_radial_velocity
    rmax = loader.rmax
    kind = loader._catname

    if kind in ["LOSS", "Foundation"]:
        keys = ["RA", "DEC", "z_CMB", "mB", "x1", "c", "e_mB", "e_x1", "e_c"]
        RA, dec, zCMB, mB, x1, c, e_mB, e_x1, e_c = (loader.cat[k] for k in keys)  # noqa
        e_zCMB = None

        mask = (zCMB < zcmb_max)
        calibration_params = {"mB": mB[mask], "x1": x1[mask], "c": c[mask],
                              "e_mB": e_mB[mask], "e_x1": e_x1[mask],
                              "e_c": e_c[mask]}

        model = PV_LogLikelihood(
            los_overdensity[:, mask], los_velocity[:, mask], rmax[:, mask],
            RA[mask], dec[mask], zCMB[mask], e_zCMB, calibration_params,
            loader.rdist, loader._Omega_m, "SN", name=kind)
    elif "Pantheon+" in kind:
        keys = ["RA", "DEC", "zCMB", "mB", "x1", "c", "biasCor_m_b", "mBERR",
                "x1ERR", "cERR", "biasCorErr_m_b", "zCMB_SN", "zCMB_Group",
                "zCMBERR"]

        RA, dec, zCMB, mB, x1, c, bias_corr_mB, e_mB, e_x1, e_c, e_bias_corr_mB, zCMB_SN, zCMB_Group, e_zCMB = (loader.cat[k] for k in keys)  # noqa
        mB -= bias_corr_mB
        e_mB = np.sqrt(e_mB**2 + e_bias_corr_mB**2)

        mask = zCMB < zcmb_max

        if kind == "Pantheon+_groups":
            mask &= np.isfinite(zCMB_Group)

        if kind == "Pantheon+_groups_zSN":
            mask &= np.isfinite(zCMB_Group)
            zCMB = zCMB_SN

        if kind == "Pantheon+_zSN":
            zCMB = zCMB_SN

        calibration_params = {"mB": mB[mask], "x1": x1[mask], "c": c[mask],
                              "e_mB": e_mB[mask], "e_x1": e_x1[mask],
                              "e_c": e_c[mask]}
        model = PV_LogLikelihood(
            los_overdensity[:, mask], los_velocity[:, mask], rmax[:, mask],
            RA[mask], dec[mask], zCMB[mask], e_zCMB[mask], calibration_params,
            loader.rdist, loader._Omega_m, "SN", name=kind)
    elif kind in ["SFI_gals", "2MTF", "SFI_gals_masked"]:
        keys = ["RA", "DEC", "z_CMB", "mag", "eta", "e_mag", "e_eta"]
        RA, dec, zCMB, mag, eta, e_mag, e_eta = (loader.cat[k] for k in keys)

        mask = (zCMB < zcmb_max)
        if kind == "SFI_gals":
            mask &= (eta > -0.15) & (eta < 0.2)
            if verbose:
                print("Emplyed eta cut for SFI galaxies.", flush=True)

        calibration_params = {"mag": mag[mask], "eta": eta[mask],
                              "e_mag": e_mag[mask], "e_eta": e_eta[mask]}
        model = PV_LogLikelihood(
            los_overdensity[:, mask], los_velocity[:, mask], rmax[:, mask],
            RA[mask], dec[mask], zCMB[mask], None, calibration_params,
            loader.rdist, loader._Omega_m, "TFR", name=kind)
    else:
        raise ValueError(f"Catalogue `{kind}` not recognized.")

    fprint(f"selected {np.sum(mask)}/{len(mask)} galaxies in catalogue `{kind}`")  # noqa

    return model


###############################################################################
#                     Predicting z_cosmo from z_obs                           #
###############################################################################


def _posterior_element(r, beta, Vext_radial, los_velocity, Omega_m, zobs,
                       sigma_v, alpha, dVdOmega, los_density):
    """
    Helper function function to compute the unnormalized posterior in
    `Observed2CosmologicalRedshift`.
    """
    zobs_pred = predict_zobs(r, beta, Vext_radial, los_velocity, Omega_m)

    # Likelihood term
    dcz = SPEED_OF_LIGHT * (zobs - zobs_pred)
    posterior = jnp.exp(-0.5 * dcz**2 / sigma_v**2)
    posterior /= jnp.sqrt(2 * jnp.pi * sigma_v**2)

    # Prior term
    posterior *= dVdOmega * los_density**alpha

    return posterior


class BaseObserved2CosmologicalRedshift(ABC):
    """Base class for `Observed2CosmologicalRedshift`."""
    def __init__(self, calibration_samples, r_xrange):
        # Check calibration samples input.
        for i, key in enumerate(calibration_samples.keys()):
            x = calibration_samples[key]
            if not isinstance(x, (np.ndarray, jnp.ndarray)):
                raise ValueError(
                    f"Calibration sample `{key}` must be an array.")

            if x.ndim != 1 and key != "Vext":
                raise ValueError(f"Calibration samples `{key}` must be 1D.")

            if i == 0:
                ncalibratrion = len(x)

            if len(x) != ncalibratrion:
                raise ValueError(
                    "Calibration samples do not have the same length.")

            calibration_samples[key] = jnp.asarray(x)

        if "alpha" not in calibration_samples:
            print("No `alpha` calibration sample found. Setting it to 1.",
                  flush=True)
            calibration_samples["alpha"] = jnp.ones(ncalibratrion)

        if "beta" not in calibration_samples:
            print("No `beta` calibration sample found. Setting it to 1.",
                  flush=True)
            calibration_samples["beta"] = jnp.ones(ncalibratrion)

        # Get the stepsize, we need it to be constant for Simpson's rule.
        dr = np.diff(r_xrange)
        if not np.all(np.isclose(dr, dr[0], atol=1e-5)):
            raise ValueError("The radial step size must be constant.")
        dr = dr[0]

        self._calibration_samples = calibration_samples
        self._ncalibration_samples = ncalibratrion

        # It is best to JIT compile the functions right here.
        self._jit_posterior_element = jit(_posterior_element)

    def get_calibration_samples(self, key):
        """Get calibration samples for a given key."""
        if key not in self._calibration_samples:
            raise ValueError(f"Key `{key}` not found in calibration samples. "
                             f"Available keys are: `{self.calibration_keys}`.")

        return self._calibration_samples[key]

    @property
    def ncalibration_samples(self):
        """Number of calibration samples."""
        return self._ncalibration_samples

    @property
    def calibration_keys(self):
        """Calibration sample keys."""
        return list(self._calibration_samples.keys())


class Observed2CosmologicalRedshift(BaseObserved2CosmologicalRedshift):
    """
    Model to predict the cosmological redshift from the observed redshift in
    the CMB frame.

    Parameters
    ----------
    calibration_samples : dict
        Dictionary of flow calibration samples (`alpha`, `beta`, `Vext`,
        `sigma_v`, ...).
    r_xrange : 1-dimensional array
        Radial comoving distances where the fields are interpolated for each
        object.
    Omega_m : float
        Matter density parameter.
    """
    def __init__(self, calibration_samples, r_xrange, Omega_m):
        super().__init__(calibration_samples, r_xrange)
        self._r_xrange = jnp.asarray(r_xrange, dtype=jnp.float32)
        self._zcos_xrange = dist2redshift(self._r_xrange, Omega_m)
        self._Omega_m = Omega_m

        # Comoving volume element with some arbitrary normalization
        dVdOmega = gradient_redshift2dist(self._zcos_xrange, Omega_m)
        # TODO: Decide about the presence of this correction.
        dVdOmega *= self._r_xrange**2
        self._dVdOmega = dVdOmega / jnp.mean(dVdOmega)

    def posterior_mean_std(self, x, px):
        """
        Calculate the mean and standard deviation of a 1-dimensional PDF.
        """
        mu = simpson(x * px, x=x)
        std = (simpson(x**2 * px, x=x) - mu**2)**0.5
        return mu, std

    def posterior_zcosmo(self, zobs, RA, dec, los_density, los_velocity,
                         extra_sigma_v=None, verbose=True):
        """
        Calculate `p(z_cosmo | z_CMB, calibration)` for a single object.

        Parameters
        ----------
        zobs : float
            Observed redshift.
        RA, dec : float
            Right ascension and declination in radians.
        los_density : 1-dimensional array
            LOS density field.
        los_velocity : 1-dimensional array
            LOS radial velocity field.
        extra_sigma_v : float, optional
            Any additional velocity uncertainty.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        zcosmo : 1-dimensional array
            Cosmological redshift at which the PDF is evaluated.
        posterior : 1-dimensional array
            Posterior PDF.
        """
        Vext = self.get_calibration_samples("Vext")
        Vext_radial = project_Vext(*[Vext[:, i] for i in range(3)], RA, dec)

        alpha = self.get_calibration_samples("alpha")
        beta = self.get_calibration_samples("beta")
        sigma_v = self.get_calibration_samples("sigma_v")

        if extra_sigma_v is not None:
            sigma_v = jnp.sqrt(sigma_v**2 + extra_sigma_v**2)

        posterior = np.zeros((self.ncalibration_samples, len(self._r_xrange)),
                             dtype=np.float32)
        for i in trange(self.ncalibration_samples, desc="Marginalizing",
                        disable=not verbose):
            posterior[i] = self._jit_posterior_element(
                self._r_xrange, beta[i], Vext_radial[i], los_velocity,
                self._Omega_m, zobs, sigma_v[i], alpha[i], self._dVdOmega,
                los_density)

        # # Normalize the posterior for each flow sample and then stack them.
        posterior /= simpson(posterior, x=self._zcos_xrange, axis=-1)[:, None]
        posterior = jnp.nanmean(posterior, axis=0)

        return self._zcos_xrange, posterior


def stack_pzosmo_over_realizations(n, obs2cosmo_models, loaders, zobs_catname,
                                   pzcosmo_kwargs={}, verbose=True):
    """
    Stack the posterior PDFs of `z_cosmo` for a given galaxy index `n` over
    multiple constrained realizations.

    Parameters
    ----------
    n : int
        Galaxy index in the loaders' catalogue.
    obs2cosmo_models : list
        List of `Observed2CosmologicalRedshift` instances per realization.
    loaders : list
        List of DataLoader instances per realization.
    zobs_catname : str
        Name of the observed redshift column in the catalogue.
    pzcosmo_kwargs : dict, optional
        Additional keyword arguments to pass to `posterior_zcosmo`.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    zcosmo : 1-dimensional array
        Cosmological redshift at which the PDF is evaluated.
    p_zcosmo : 1-dimensional array
        Stacked posterior PDF.
    """
    # Do some standard checks of inputs
    if not isinstance(obs2cosmo_models, list):
        raise ValueError("`obs2cosmo_models` 1must be a list.")
    if not isinstance(loaders, list):
        raise ValueError("`loaders` must be a list.")
    if len(obs2cosmo_models) != len(loaders):
        raise ValueError("The number of models and loaders must be equal.")

    for i in trange(len(obs2cosmo_models), desc="Stacking",
                    disable=not verbose):
        zobs = loaders[i].cat[zobs_catname][n]
        RA = np.deg2rad(loaders[i].cat["RA"][n])
        dec = np.deg2rad(loaders[i].cat["DEC"][n])
        los_density = loaders[i].los_density[n]
        los_velocity = loaders[i].los_radial_velocity[n]

        x, y = obs2cosmo_models[i].posterior_zcosmo(
            zobs, RA, dec, los_density, los_velocity, verbose=False,
            **pzcosmo_kwargs)

        if i == 0:
            zcosmo = x
            p_zcosmo = np.empty((len(loaders), len(x)), dtype=np.float32)

        p_zcosmo[i] = y

    # Stack the posterior PDFs
    p_zcosmo = np.nanmean(p_zcosmo, axis=0)

    return zcosmo, p_zcosmo
