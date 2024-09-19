# Copyright (C) 2023 Richard Stiskalek
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
Script to construct the density and velocity fields for a simulation snapshot.
The SPH filter is implemented in the cosmotool package.
"""
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from os import remove
from os.path import exists, join

import hdf5plugin  # noqa
import readgadget
import numpy as np
from h5py import File


def now():
    return datetime.now()


def generate_unique_id(file_path):
    """
    Generate a unique ID for a file path.
    """
    return file_path.replace('/', '_').replace(':', '_')


def prepare_random(temporary_output_path, npart=100, dtype=np.float32):
    """
    Prepare a random dataset for the SPH filter.
    """
    print("Preparing random dataset.", flush=True)
    arr = np.full((npart, 7), np.nan, dtype=dtype)

    arr[:, :3] = np.random.uniform(0, 1, (npart, 3))
    arr[:, 3:6] = np.random.normal(0, 1, (npart, 3))
    arr[:, 6] = np.ones(npart, dtype=dtype)

    dset = np.random.random((npart, 7)).astype(dtype)
    dset[:, 6] = np.ones(npart, dtype=dtype)

    with File(temporary_output_path, 'w') as target:
        target.create_dataset("particles", data=dset, dtype=dtype)

    return 1.


def prepare_gadget(snapshot_path, temporary_output_path):
    """
    Prepare a GADGET snapshot for the SPH filter. Assumes there is only a
    single file per snapshot.
    """
    with File(snapshot_path, 'r') as source, File(temporary_output_path, 'w') as target:  # noqa
        boxsize = source["Header"].attrs["BoxSize"]

        npart = sum(source["Header"].attrs["NumPart_Total"])
        nhighres = source["Header"].attrs["NumPart_Total"][1]

        dset = target.create_dataset("particles", (npart, 7), dtype=np.float32)

        # Copy to this dataset the high-resolution particles.
        dset[:nhighres, :3] = source["PartType1/Coordinates"][:]
        dset[:nhighres, 3:6] = source["PartType1/Velocities"][:]
        dset[:nhighres, 6] = np.ones(nhighres, dtype=np.float32) * source["Header"].attrs["MassTable"][1]  # noqa

        # Now copy the low-resolution particles.
        dset[nhighres:, :3] = source["PartType5/Coordinates"][:]
        dset[nhighres:, 3:6] = source["PartType5/Velocities"][:]
        dset[nhighres:, 6] = source["PartType5/Masses"][:]

    return boxsize


def prepara_gadget2(snapshot_path, temporary_output_path):
    ptype = [1]
    header = readgadget.header(snapshot_path)

    boxsize = header.boxsize / 1000
    npart = header.nall[1]
    mpart = header.massarr[1] * 1e10

    print(f"{'Boxsize':<20}: {boxsize}")
    print(f"{'Npart':<20}: {npart}")
    print(f"{'Mpart':<20}: {mpart}")

    with File(temporary_output_path, 'w') as target:
        dset = target.create_dataset("particles", (npart, 7), dtype=np.float32)

        dset[:, :3] = readgadget.read_block(snapshot_path, "POS ", ptype) / 1000  # noqa
        dset[:, 3:6] = readgadget.read_block(snapshot_path, "VEL ", ptype)
        dset[:, 6] = np.ones(npart, dtype=np.float32) * mpart

    return boxsize


def prepare_swift(snapshot_path, temporary_output_path):
    """
    Read in SWIFT Manticore files from Stuart.
    """
    # Get some basic information about the snapshots.
    with File(snapshot_path, 'r') as src, File(temporary_output_path, 'w') as target: # noqa
        h = src["Cosmology"].attrs["h"][0]
        npart = src["Header"].attrs["NumPart_Total"][1]

        # Convert to Msun / h
        mpart = src["Header"].attrs["InitialMassTable"][1] * 1e10 * h
        # Convert to Mpc / h
        boxsize = src["Header"].attrs["BoxSize"][0] * h

        print(f"{'Boxsize':<20}: {boxsize}")
        print(f"{'Npart':<20}: {npart}")
        print(f"{'Mpart':<20}: {mpart}")

        dset = target.create_dataset("particles", (npart, 7), dtype=np.float32)

        dset[:, :3] = src["DMParticles/Coordinates"][:] * h  # Mpc / h
        dset[:, 3:6] = src["DMParticles/Velocities"][:]      # km/s
        dset[:, 6] = np.ones(npart, dtype=np.float32) * mpart

    return boxsize


def run_sph_filter(particles_path, output_path, boxsize, resolution,
                   SPH_executable):
    """
    Run the SPH filter on a snapshot.
    """
    if not exists(particles_path):
        raise RuntimeError(f"Particles file `{particles_path}` does not exist.")  # noqa
    if not isinstance(boxsize, (int, float)):
        raise TypeError("`boxsize` must be a number.")
    if not isinstance(resolution, int):
        raise TypeError("`resolution` must be an integer.")
    if not exists(SPH_executable):
        raise RuntimeError(f"SPH executable `{SPH_executable}` does not exist.")  # noqa

    command = [SPH_executable, particles_path, str(1e14), str(boxsize),
               str(resolution), str(0), str(0), str(0), output_path, "1"]
    print(f"{now()}: executing `simple3DFilter`.", flush=True)
    start_time = now()
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True)

    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
    process.wait()

    if process.returncode != 0:
        raise RuntimeError("`simple3DFilter`failed.")
    else:
        dt = now() - start_time
        print(f"{now()}: `simple3DFilter`completed successfully in {dt}.",
              flush=True)


def main(snapshot_path, output_path, resolution, scratch_space, SPH_executable,
         snapshot_kind):
    """
    Construct the density and velocity fields for a simulation snapshot using
    `cosmotool` [1].

    Parameters
    ----------
    snapshot_path : str
        Path to the simulation snapshot.
    output_path : str
        Path to the output HDF5 file.
    resolution : int
        Resolution of the density field.
    scratch_space : str
        Path to a folder where temporary files can be stored.
    SPH_executable : str
        Path to the `simple3DFilter` executable [1].
    snapshot_kind : str
        Kind of the simulation snapshot.

    Returns
    -------
    None

    References
    ----------
    [1] https://bitbucket.org/glavaux/cosmotool/src/master/sample/simple3DFilter.cpp  # noqa
    """
    # First get the temporary file path.
    if snapshot_kind in ["gadget2", "gadget4", "swift"]:
        temporary_output_path = join(
            scratch_space, generate_unique_id(snapshot_path))
    elif snapshot_kind == "ramses":
        temporary_output_path = snapshot_path
    else:
        raise NotImplementedError("Only GADGET HDF5 or preprocessed RAMSES "
                                  "snapshots are supported.")

    if not temporary_output_path.endswith(".hdf5"):
        temporary_output_path += ".hdf5"

    # Print the job information.
    print()
    print("---------- SPH Density & Velocity Field Job Information ----------")
    print(f"Snapshot path:     {snapshot_path}")
    print(f"Output path:       {output_path}")
    print(f"Temporary path:    {temporary_output_path}")
    print(f"Resolution:        {resolution}")
    print(f"Scratch space:     {scratch_space}")
    print(f"SPH executable:    {SPH_executable}")
    print(f"Snapshot kind:     {snapshot_kind}")
    print("------------------------------------------------------------------")
    print(flush=True)

    # Prepare or read-off the temporary snapshot file.
    if snapshot_kind == "gadget4":
        print(f"{now()}: preparing snapshot...", flush=True)
        boxsize = prepare_gadget(snapshot_path, temporary_output_path)
        print(f"{now()}: wrote temporary data to {temporary_output_path}.",
              flush=True)
    elif snapshot_kind == "gadget2":
        print(f"{now()}: preparing snapshot...", flush=True)
        boxsize = prepara_gadget2(snapshot_path, temporary_output_path)
        print(f"{now()}: wrote temporary data to {temporary_output_path}.")
    elif snapshot_kind == "swift":
        print(f"{now()}: preparing snapshot...", flush=True)
        boxsize = prepare_swift(snapshot_path, temporary_output_path)
        print(f"{now()}: wrote temporary data to {temporary_output_path}.")
    else:
        raise RuntimeError(f"Snapshot kind `{snapshot_kind}` not implemented.")

    # Run the SPH filter.
    run_sph_filter(temporary_output_path, output_path, boxsize, resolution,
                   SPH_executable)

    # Remove the temporary snapshot file if it was created.
    if snapshot_kind == "gadget4":
        print(f"{now()}: removing the temporary snapshot file.", flush=True)
        try:
            remove(temporary_output_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate SPH density and velocity field.")  # noqa
    parser.add_argument("--snapshot_path", type=str, required=True,
                        help="Path to the simulation snapshot.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output HDF5 file.")
    parser.add_argument("--resolution", type=int, required=True,
                        help="Resolution of the density and velocity field.")
    parser.add_argument("--scratch_space", type=str, required=True,
                        help="Path to a folder where temporary files can be stored.")  # noqa
    parser.add_argument("--SPH_executable", type=str, required=True,
                        help="Path to the `simple3DFilter` executable.")
    parser.add_argument("--snapshot_kind", type=str, required=True,
                        choices=["gadget4", "gadget2", "ramses", "swift"],
                        help="Kind of the simulation snapshot.")
    args = parser.parse_args()

    if not args.output_path.endswith(".hdf5"):
        raise RuntimeError("Output path must end with `.hdf5`.")

    main(args.snapshot_path, args.output_path, args.resolution,
         args.scratch_space, args.SPH_executable, args.snapshot_kind)
