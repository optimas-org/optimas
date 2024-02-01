"""Script for simulating an LPA with ionization injection with FBPIC."""

import numpy as np
from scipy.constants import c, e, m_e, m_p

from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.openpmd_diag import BoostedParticleDiagnostic
from fbpic.lpa_utils.boosted_frame import BoostConverter


def LUXlaser(
    energy_measured_joule,
    FWHM_x_um,
    FWHM_t_fs,
    lambda_laser=0.8,
    T_beamline=1,
    focus_factor=1,
    temporal_factor=1,
):
    """Get the laser peak intensity, a0, spot size and length."""
    energy_gauss_joule = (
        energy_measured_joule * T_beamline * focus_factor * temporal_factor
    )
    FWHMtoSigma = 2 * np.sqrt(2 * np.log(2))
    I_0 = energy_gauss_joule / (
        (np.sqrt(2 * np.pi)) ** 3
        * (FWHM_x_um * 1e-4 / FWHMtoSigma) ** 2
        * (FWHM_t_fs * 1e-15 / FWHMtoSigma)
    )
    a_0 = 8.5492970742069339e-10 * lambda_laser * np.sqrt(I_0)
    w_0 = FWHM_x_um * 2 / FWHMtoSigma
    c_tau = 2.0 * (FWHM_t_fs * 1e-15) / FWHMtoSigma * 299792458.0 * 1e6
    return I_0, a_0, w_0, c_tau


# Optimization parameters.
laser_scale = {{laser_scale}}
z_foc = {{z_foc}} * 1.0e-3
mult = {{mult}}
plasma_scale = {{plasma_scale}}
spot_scale = 1.0
t_scale = 1.0


# Plasma density profile parameters.
dens_z = np.linspace(0, 8e-3, 1000)
dens_h2 = plasma_scale * (
    5e23 * np.exp(-(((dens_z - 2.9e-3) / 1.0e-3) ** 2))
    + 6e23 * np.exp(-(((dens_z - 5.3e-3) / 1.7e-3) ** 4))
)
dens_n2 = plasma_scale * (0.5e23 * np.exp(-(((dens_z - 2.5e-3) / 0.5e-3) ** 2)))

dens_h2 = dens_h2 - (mult - 1) * dens_n2
dens_n2 = dens_n2 * mult


# The simulation box
Nz = 3200  # Number of gridpoints along z
zmax = 0.0e-6  # Right end of the simulation box (meters)
zmin = -80.0e-6  # Left end of the simulation box (meters)
Nr = 270  # Number of gridpoints along r
rmax = 135.0e-6  # Length of the box along r (meters)
Nm = 2  # Number of modes used


# Boost factor and converter.
gamma_boost = 5.0
boost = BoostConverter(gamma_boost)


# Maximum simulation length
Lmax = np.amax(dens_z + zmax - zmin)


# The simulation timestep (seconds)
dt = min(rmax / (2 * gamma_boost * Nr), (zmax - zmin) / Nz / c)


# Order of the field solver.
n_order = 32


# Whether to use the GPU.
use_cuda = True


# Plasma particles.
p_zmin = 0.0e-6  # Position of the beginning of the plasma (meters)
p_rmax = 135.0e-6

# Particles per cell (Hydrogen)
p_nz = 1  # Number of particles per cell along z
p_nr = 2  # Number of particles per cell along r
p_nt = 4  # Number of particles per cell along theta

# Particles per cell (Nitrogen)
p_nz_N = 2
p_nr_N = 2
p_nt_N = 4


# Laser parameters and profile.
e_l = 2.56 * laser_scale  # Energy Joule
w_l = 25.0 * spot_scale  # Width (intensity) FWHM mu
w0_flat = w_l * 1.0e-6 / 1.609  # Flat-top w0 from FWHM for N=100
tau_l = 34.0 * t_scale  # Duration (intensity) FWHM fs
I0, a0, w0, ctau = LUXlaser(e_l, w_l, tau_l)
w0 *= 1.0e-6
ctau *= 1.0e-6
z0 = -3 * ctau  # Laser centroid
laser_profile = GaussianLaser(a0, w0, ctau / c, z0, zf=z_foc)


# Plasma density functions.
def dens_func_H(z, r):
    """Hydrogen density function."""
    z_lab = z * gamma_boost
    return 2 * np.interp(z_lab, dens_z, dens_h2)


def dens_func_N(z, r):
    """Nitrogen density function."""
    z_lab = z * gamma_boost
    return 2 * np.interp(z_lab, dens_z, dens_n2)


def dens_func_e(z, r):
    """Electron density function."""
    return dens_func_H(z, r) + 5 * dens_func_N(z, r)


# The moving window
v_window = c


# Velocity of the Galilean frame (for suppression of the NCI)
v_comoving = -np.sqrt(gamma_boost**2 - 1.0) / gamma_boost * c


# The diagnostics
diag_period = 100  # Period of the diagnostics in number of timesteps
# Whether to write the fields in the lab frame
Ntot_snapshot_lab = 2
dt_snapshot_lab = (Lmax + (zmax - zmin)) / v_window / (Ntot_snapshot_lab - 1)
track_bunch = False  # Whether to tag and track the particles of the bunch


# The interaction length (meters) and time (seconds) of the simulation.
# (i.e. the time it takes for the moving window to slide across the plasma)
L_interact = Lmax  # the plasma length
T_interact = boost.interaction_time(L_interact, (zmax - zmin), v_window)


# Carrying out the simulation
if __name__ == "__main__":
    # Initialize the simulation object
    sim = Simulation(
        Nz,
        zmax,
        Nr,
        rmax,
        Nm,
        dt,
        zmin=zmin,
        boundaries={"z": "open", "r": "open"},
        initialize_ions=False,
        n_order=n_order,
        use_cuda=use_cuda,
        v_comoving=v_comoving,
        gamma_boost=gamma_boost,
        verbose_level=2,
        particle_shape="cubic",
        use_galilean=True,
    )

    # By default the simulation initializes an electron species (sim.ptcl[0])
    # Because we did not pass the arguments `n`, `p_nz`, `p_nr`, `p_nz`,
    # this electron species does not contain any macroparticles.
    # It is okay to just remove it from the list of species.
    sim.ptcl = []

    # Add the Helium ions (pre-ionized up to level 1),
    # the Nitrogen ions (pre-ionized up to level 5)
    # and the associated electrons (from the pre-ionized levels)
    atoms_N = sim.add_new_species(
        q=5 * e,
        m=14.0 * m_p,
        n=1,
        dens_func=dens_func_N,
        p_nz=p_nz_N,
        p_nr=p_nr_N,
        p_nt=p_nt_N,
        p_zmin=p_zmin,
        p_rmax=p_rmax,
    )
    atoms_H = sim.add_new_species(
        q=e,
        m=1 * m_p,
        n=1,
        dens_func=dens_func_H,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
        p_zmin=p_zmin,
        p_rmax=p_rmax,
    )
    elec = sim.add_new_species(
        q=-e,
        m=m_e,
        n=1,
        dens_func=dens_func_e,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
        p_zmin=p_zmin,
        p_rmax=p_rmax,
    )

    # Activate ionization of N ions (for levels above 5).
    # Store the created electrons in a new dedicated electron species that
    # does not contain any macroparticles initially
    elec_from_N = sim.add_new_species(q=-e, m=m_e)
    atoms_N.make_ionizable("N", target_species=elec_from_N, level_start=5)

    # Add a laser to the fields of the simulation
    add_laser_pulse(
        sim,
        laser_profile,
        gamma_boost=gamma_boost,
        method="antenna",
        z0_antenna=0,
    )

    # Convert parameter to boosted frame
    (v_window,) = boost.velocity([v_window])

    # Configure the moving window
    sim.set_moving_window(v=v_window)

    # Add a diagnostics
    write_dir = "diags"
    sim.diags = [
        BoostedParticleDiagnostic(
            zmin,
            zmax,
            c,
            dt_snapshot_lab,
            Ntot_snapshot_lab,
            gamma_boost,
            diag_period,
            sim.fld,
            species={"electrons from N": elec_from_N},
            comm=sim.comm,
            write_dir=write_dir,
        )
    ]

    # Remove step 0 outputs
    sim.diags[0].snapshots.pop(0)

    # Calculate number of simulation steps to perform.
    N_step = int(T_interact / sim.dt)

    # Run the simulation
    sim.step(N_step)
