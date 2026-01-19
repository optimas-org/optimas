"""Script for simulating an LPA with external injection with Wake-T and FBPIC.

The simulation code is determined from the `task` parameter. The beam
current, position and length are parameters exposed to the optimizer to
try to achieve optimal beam loading.
"""

import numpy as np
import scipy.constants as ct
from wake_t import GaussianPulse, PlasmaStage, ParticleBunch
import aptools.plasma_accel.general_equations as ge
from bunch_utils import trapezoidal_bunch

# Parammeters exposed to optimizer.
task = {{task}}
beam_i_1 = {{beam_i_1}}
beam_i_2 = {{beam_i_2}}
beam_z_i_2 = {{beam_z_i_2}}
beam_length = {{beam_length}}


# General simulation parameters.
n_p_plateau = 2e23
l_plateau = 10e-2
w0_laser = 40e-6
z_beam = (50 + beam_z_i_2) * 1e-6
l_beam = beam_length * 1e-6
i1_beam = beam_i_1 * 1e3
i2_beam = beam_i_2 * 1e3


def run_simulation():
    """Run a simulation of the LPA with Wake-T or FBPIC."""
    # Base laser parameters.
    E_laser = 10  # J
    tau_laser = 25e-15  # s (fwhm)
    lambda0 = 0.8e-6  # m
    a0 = determine_laser_a0(E_laser, tau_laser, w0_laser, lambda0)

    # Base beam parameters.
    E_beam = 200  # MeV
    gamma_beam = E_beam / 0.511
    n_emitt_x = 3e-6
    n_emitt_y = 0.5e-6
    ene_sp = 0.1  # %
    n_part = 1e5
    sz0 = 1e-6  # gaussian decay
    kp = np.sqrt(n_p_plateau * ct.e**2 / (ct.epsilon_0 * ct.m_e * ct.c**2))
    kbeta = kp / np.sqrt(2.0 * gamma_beam)  # betatron wavenumber (blowout)
    betax0 = 1.0 / kbeta  # matched beta
    sx0 = np.sqrt(n_emitt_x * betax0 / gamma_beam)  # matched beam size (rms)
    sy0 = np.sqrt(n_emitt_y * betax0 / gamma_beam)  # matched beam size (rms)

    # Determine guiding channel.
    r_e = ct.e**2 / (4.0 * np.pi * ct.epsilon_0 * ct.m_e * ct.c**2)
    rel_delta_n_over_w2 = 1.0 / (np.pi * r_e * w0_laser**4 * n_p_plateau)

    # Generate bunch
    x, y, z, ux, uy, uz, q = trapezoidal_bunch(
        i1_beam,
        i2_beam,
        n_part=n_part,
        gamma0=gamma_beam,
        s_g=ene_sp * gamma_beam / 100,
        length=l_beam,
        s_z=sz0,
        emit_x=n_emitt_x,
        s_x=sx0,
        emit_y=n_emitt_y,
        s_y=sy0,
        zf=0.0,
        tf=0.0,
    )
    z -= l_beam / 2 + z_beam
    w = np.abs(q / ct.e)
    bunch = ParticleBunch(w, x, y, z, ux, uy, uz, name="bunch")

    # Distance between right boundary and laser centroid.
    dz_lb = 4.0 * ct.c * tau_laser

    # Maximum radial extension of the plasma.
    p_rmax = 2.5 * w0_laser

    # Box length.
    l_box = dz_lb + 90e-6

    # Number of diagnostics
    n_out = 3

    if task == "fbpic":
        run_fbpic(
            a0,
            w0_laser,
            tau_laser,
            lambda0,
            bunch,
            n_p_plateau,
            l_plateau,
            rel_delta_n_over_w2,
            p_rmax,
            dz_lb,
            l_box,
            n_out,
        )
    elif task == "wake-t":
        run_wake_t(
            a0,
            w0_laser,
            tau_laser,
            lambda0,
            bunch,
            n_p_plateau,
            l_plateau,
            rel_delta_n_over_w2,
            p_rmax,
            dz_lb,
            l_box,
            n_out - 1,
        )


def determine_laser_a0(ene, tau_fwhm, w0, lambda0):
    """Determine the laser a0 from the energy, size and wavelength."""
    tau = tau_fwhm / np.sqrt(2.0 * np.log(2))
    k0 = 2.0 * np.pi / lambda0  # Laser wavenumber
    PA = ct.epsilon_0 * ct.c**5 * ct.m_e**2 / ct.e**2  # Power constant
    P0 = ene / (np.sqrt(2 * np.pi) * (tau / 2))
    i0 = P0 / ((np.pi / 2) * w0**2)
    a0 = np.sqrt(i0 / (PA * k0**2 / 2))
    return a0


def density_profile(z):
    """Define the longitudinal density profile of the plasma."""
    # Allocate relative density
    n = np.ones_like(z)
    # Make zero before plateau
    n = np.where(z < 0, 0, n)
    # Make zero after plateau
    n = np.where(z >= l_plateau, 0, n)
    # Return absolute density
    return n * n_p_plateau


def run_wake_t(
    a0,
    w0,
    tau_fwhm,
    lambda0,
    bunch,
    n_p,
    l_plasma,
    pc,
    p_rmax,
    dz_lb,
    l_box,
    n_out,
):
    """Run a Wake-T simulation of the LPA."""
    # Create laser.
    laser = GaussianPulse(
        xi_c=0.0, l_0=lambda0, w_0=w0, a_0=a0, tau=tau_fwhm, z_foc=0.0
    )

    # Plasma stage.
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    dr = s_d / 20
    dz = tau_fwhm * ct.c / 40
    r_max = w0 * 4
    plasma = PlasmaStage(
        l_plasma,
        density=density_profile,
        wakefield_model="quasistatic_2d",
        n_out=n_out,
        laser=laser,
        laser_evolution=True,
        r_max=r_max,
        r_max_plasma=p_rmax,
        xi_min=dz_lb - l_box,
        xi_max=dz_lb,
        n_r=int(r_max / dr),
        n_xi=int(l_box / dz),
        dz_fields=l_box * 2,
        ppc=4,
        parabolic_coefficient=pc,
        max_gamma=25,
        dt_bunch=calculate_waket_timestep(bunch, n_p),
        bunch_pusher="boris",
    )

    # Do tracking.
    plasma.track(bunch, opmd_diag=True, diag_dir="diags")


def run_fbpic(
    a0,
    w0,
    tau_fwhm,
    lambda0,
    bunch,
    n_p,
    l_plasma,
    pc,
    p_rmax,
    dz_lb,
    l_box,
    n_out,
):
    """Run an FBPIC simulation of the LPA."""
    from fbpic.main import Simulation
    from fbpic.lpa_utils.boosted_frame import BoostConverter
    from fbpic.lpa_utils.bunch import add_particle_bunch_from_arrays
    from fbpic.lpa_utils.laser import add_laser
    from custom_ptcl_diags import BackTransformedParticleDiagnostic

    use_cuda = True
    n_order = -1

    # Boosted frame
    gamma_boost = 25.0
    boost = BoostConverter(gamma_boost)

    # The laser (Gaussian)
    lambda0 = 0.8e-6  # Laser wavelength
    # Laser duration (2 sigmas) in intensity
    tau = tau_fwhm / np.sqrt(2.0 * np.log(2))
    ctau = tau * ct.c

    # The simulation box
    zmin = -l_box  # Left  edge of the simulation box (meters)
    zmax = 0.0e-6  # Right edge of the simulation box (meters)
    rmax = w0 * 4
    dz_adv = lambda0 / 80.0  # Advised longitudinal resolution
    Nz_adv = int(l_box / dz_adv)
    Nz = Nz_adv  # Number of gridpoints along z
    Nm = 3  # Number of modes used
    s_d = ge.plasma_skin_depth(n_p * 1e-6)
    dr = s_d / 20
    Nr = int(rmax / dr)

    # Laser centroid
    z0 = zmax - dz_lb

    # The simulation timestep
    dz = (zmax - zmin) / Nz
    dt = dz / ct.c

    # The moving window
    v_window = ct.c  # velocity of the window

    # Velocity of the Galilean frame (for suppression of the NCI)
    (v_comoving,) = boost.velocity([0.0])

    # ------------

    # The plasma particles
    p_zmin = zmax  # Position of the beginning of the plasma (meters)
    p_nz = 2  # Number of particles per cell along z
    p_nr = 2  # Number of particles per cell along r
    p_nt = 8  # Number of particles per cell along theta

    # The interaction length of the simulation (meters)
    L_lab_interact = l_plasma

    # Duration of plasma interaction (i.e. the time it takes for the moving
    # window to slide across the plasma)
    T_lab_interact_plasma = L_lab_interact / v_window

    # Number of discrete diagnostic snapshots in the lab frame
    N_lab_diag = n_out

    # data dumping period in dt units:
    dt_lab_diag_period = T_lab_interact_plasma / (
        N_lab_diag - 1
    )  # Period of the diagnostics (seconds)

    # In boosted frame:
    (v_window_boosted,) = boost.velocity([v_window])

    # Interaction time in boosted frame
    T_interact = boost.interaction_time(L_lab_interact, (zmax - zmin), v_window)

    # Period of writing the cached backtransformed lab frame diagnostics to
    # disk (in number of iterations)
    write_period = 200

    # Density function
    def dens_func(z, r):
        z_lab = z * gamma_boost
        n = density_profile(z_lab) / n_p
        n = n * (1.0 + pc * r**2)
        return n

    # External bunch
    if bunch is not None:
        x, y, z, px, py, pz, w = (
            bunch.x,
            bunch.y,
            bunch.xi,
            bunch.px,
            bunch.py,
            bunch.pz,
            bunch.w,
        )
        z += z0

    # Initialize the simulation object
    sim = Simulation(
        Nz=Nz,
        zmax=zmax,
        Nr=Nr,
        rmax=rmax,
        Nm=Nm,
        dt=dt,
        zmin=zmin,
        v_comoving=v_comoving,
        gamma_boost=boost.gamma0,
        n_order=n_order,
        use_cuda=use_cuda,
        boundaries={"z": "open", "r": "reflective"},
        particle_shape="cubic",
    )

    # Add the Helium ions (full pre-ionized: levels 1 and 2)
    sim.add_new_species(
        q=ct.e,
        m=ct.m_p,
        n=n_p,
        dens_func=dens_func,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
        p_zmin=p_zmin,
        p_rmax=p_rmax,
    )

    # Plasma electrons: coming from helium
    sim.add_new_species(
        q=-ct.e,
        m=ct.m_e,
        n=n_p,
        dens_func=dens_func,
        p_nz=p_nz,
        p_nr=p_nr,
        p_nt=p_nt,
        p_zmin=p_zmin,
        p_rmax=p_rmax,
    )

    # Add an electron bunch
    if bunch is not None:
        add_particle_bunch_from_arrays(
            sim=sim,
            q=-ct.e,
            m=ct.m_e,
            x=x,
            y=y,
            z=z,
            ux=px,
            uy=py,
            uz=pz,
            w=w,
            boost=boost,
            z_injection_plane=0.0,
        )

    # Add a laser to the fields of the simulation
    add_laser(
        sim=sim,
        a0=a0,
        w0=w0,
        ctau=ctau,
        z0=z0,
        lambda0=lambda0,
        zf=0.0,
        gamma_boost=boost.gamma0,
        method="antenna",
        z0_antenna=0.0,
        cep_phase=np.pi,
    )

    # Configure the moving window
    sim.set_moving_window(v=v_window_boosted)

    # Add diagnostics
    write_dir = "diags"

    # Set start time of diagnostics to the exact moment the bunch enters the
    # plasma. (Required to have output at same location as Wake-T)
    if bunch is not None:
        T_start_lab = (zmax - np.average(z)) / v_window
    else:
        T_start_lab = 0.0

    # Add diagnostics.
    sim.diags = []
    # sim.diags = [
    #     BackTransformedFieldDiagnostic(
    #         zmin,
    #         zmax,
    #         v_window,
    #         T_start_lab,
    #         dt_lab_diag_period,
    #         N_lab_diag,
    #         boost.gamma0,
    #         fieldtypes=["E", "B", "rho"],
    #         period=write_period,
    #         fldobject=sim.fld,
    #         comm=sim.comm,
    #         write_dir=write_dir,
    #     )
    # ]
    if bunch is not None:
        sim.diags += [
            BackTransformedParticleDiagnostic(
                zmin_lab=zmin,
                zmax_lab=zmax,
                v_lab=v_window,
                t_start_lab=T_start_lab,
                dt_snapshots_lab=dt_lab_diag_period,
                Ntot_snapshots_lab=N_lab_diag,
                gamma_boost=boost.gamma0,
                period=write_period,
                fldobject=sim.fld,
                species={"bunch": sim.ptcl[2]},
                comm=sim.comm,
                write_dir=write_dir,
            )
        ]

    # Number of iterations to perform
    N_step = int(T_interact / sim.dt) + write_period

    # Run the simulation
    sim.step(N_step)
    print("")


def calculate_waket_timestep(beam, n_p):
    """Calculate the timestep of the bunch pusher in Wake-T."""
    mean_gamma = np.sqrt(np.average(beam.pz) ** 2 + 1)
    # calculate maximum focusing along stage.
    w_p = np.sqrt(n_p * ct.e**2 / (ct.m_e * ct.epsilon_0))
    max_kx = (ct.m_e / (2 * ct.e * ct.c)) * w_p**2
    w_x = np.sqrt(ct.e * ct.c / ct.m_e * max_kx / mean_gamma)
    period_x = 1 / w_x
    dt = 0.1 * period_x
    return dt


if __name__ == "__main__":
    run_simulation()
