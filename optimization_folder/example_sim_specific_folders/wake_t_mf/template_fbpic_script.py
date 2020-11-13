"""
This example optimizes the focusing gradient of a plasma lens using
a discrete fidelity space. One fidelity runs a cheap Wake-T
simulation while the other runs a full FBPIC simulation.
"""

import numpy as np
import scipy.constants as ct

# Wake-T imports.
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_twiss
from wake_t.beamline_elements import PlasmaLens

# FBPIC imports.
from fbpic.main import Simulation
from fbpic.openpmd_diag import BoostedFieldDiagnostic, BoostedParticleDiagnostic
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.bunch import add_particle_bunch_from_arrays
from fbpic.lpa_utils.external_fields import ExternalField

# Anaylisis.
from aptools.data_analysis.beam_diagnostics import twiss_parameters
from aptools.data_handling.reading import read_beam


def run_simulation(g_lens, model):
    # Generate particle bunch.
    g_x = 600
    b_x = 1
    a_x = -np.sqrt(b_x*g_x - 1)
    em_x = 1e-6  # m
    gamma_avg = 300 / 0.511
    ene_sp = 1  # %
    Q = 10  # pC
    s_t = 10  # fs
    n_part = 1e4
    bunch = get_gaussian_bunch_from_twiss(
        em_x, em_x, a_x, a_x, b_x, b_x, gamma_avg, ene_sp, s_t, 0, q_tot=Q,
        n_part=n_part)

    if model == "wake-t":
        run_wake_t(bunch, g_lens)
    elif model == "fbpic":
        run_fbpic(bunch, g_lens)


def run_wake_t(bunch, g_lens):
    # Define plasma lens.
    p_lens = PlasmaLens(3e-2, g_lens, n_out=2)

    # Perform tracking.
    p_lens.track(bunch)

    # Analyze bunch.
    a_x, b_x, g_x = twiss_parameters(bunch.x, bunch.px, bunch.pz, w=bunch.q)

    # Save parameter to file for `analysis_script.py`.
    file_name = 'a_x_abs-{:0.3f}.out'.format(g_lens)
    np.savetxt(file_name, np.array([np.abs(a_x)]))


def run_fbpic(bunch, g_lens):
    bunch.x, bunch.y, bunch.xi, bunch.px, bunch.py, bunch.pz, bunch.q
    w = np.abs(bunch.q / ct.e)

    # ----------
    # Parameters
    # ----------
    use_cuda = True

    # The simulation box
    zmax = 0.e-6     # Length of the box along z (meters)
    zmin = -100.e-6
    rmax = 800.e-6   # Length of the box along r (meters)

    # Boosted frame
    gamma_boost = 8.

    # Order of the stencil for z derivatives in the Maxwell solver.
    n_order = -1

    # Boosted frame converter
    boost = BoostConverter(gamma_boost)

    plateau = 3.e-2

    # The particles of the plasma
    p_zmin = 0.e-6   # Position of the beginning of the plasma (meters)
    p_zmax = plateau
    p_rmin = 0.      # Minimal radial position of the plasma (meters)
    p_rmax = 700.e-6 # Maximal radial position of the plasma (meters)
    n_e = 1.e22      # The density in the labframe (electrons.meters^-3)
    p_nz = 4         # Number of particles per cell along z
    p_nr = 2         # Number of particles per cell along r
    p_nt = 6         # Number of particles per cell along theta
    uz_m = 0.        # Initial momentum of the electrons in the lab frame

    # Grid settings
    dz = 1e-6/2
    Nz = int((zmax-zmin)/dz) # Number of gridpoints along z
    dr = 5e-6
    Nr = int(rmax/dr)# Number of gridpoints along r
    Nm = 2           # Number of modes used

    # The simulation timestep
    dt = min( rmax/(2*gamma_boost*Nr), (zmax-zmin)/Nz/ct.c )  # seconds

    # The moving window (moves with the group velocity in a plasma)
    v_window = ct.c
    # Convert parameter to boosted frame
    v_window, = boost.velocity( [ v_window ] )

    # Velocity of the Galilean frame (for suppression of the NCI)
    # v_comoving = - ct.c * np.sqrt( 1. - 1./gamma_boost**2 )
    v_comoving, = boost.velocity([0.])

    # data dumping period in dt units:
    DT = np.rint(1e-3 / (dt*ct.c))
    dt_lab_diag_period = DT * dt   # Period of the diagnostics (seconds)

    # In lab frame:
    L_plasma = plateau
    # The interaction length of the simulation (meters)
    L_lab_interact = (zmax-zmin) + L_plasma + 2e-3
    # Interaction time (seconds) (to calculate number of PIC iterations)
    T_lab_interact = (L_lab_interact + (zmax - zmin)) / v_window
    # (i.e. the time it takes for the moving window to slide across the plasma)

    # Number of discrete diagnostic snapshots in the lab frame
    N_lab_diag = int(T_lab_interact / dt_lab_diag_period)

    # In boosted frame:
    L_box = zmax-zmin
    L_box_boost, dz_boost = boost.copropag_length([L_box, dz], beta_object=v_window / ct.c)
    dt_boost = dz_boost / ct.c
    v_window_boosted, = boost.velocity([v_window])
    L_plasma_boost, = boost.static_length([L_plasma])

    # Interaction time in boosted frame
    T_interact = boost.interaction_time( L_lab_interact, (zmax - zmin), v_window)

    # Period of writing the cached backtransformed lab frame diagnostics to disk
    # (in number of iterations)
    write_period = 200

    track_bunch = False  # Whether to tag and track the particles of the bunch
    
    # Convert parameters to boosted frame
    p_zmin_boost, plateau = \
        boost.static_length( [ p_zmin, plateau ] )

    # Define the density function
    def dens_func( z, r ):
        """
        User-defined function: density profile of the plasma

        It should return the relative density with respect to n_plasma,
        at the position x, y, z (i.e. return a number between 0 and 1)

        Parameters
        ----------
        z, r: 1darrays of floats
            Arrays with one element per macroparticle
        Returns
        -------
        n : 1d array of floats
            Array of relative density, with one element per macroparticles
        """
        # Allocate relative density
        n = np.ones_like(z)
        n = np.where( (z < p_zmin_boost) | ( z >= plateau), 0., n)
        return(n)

    # Define external magnetic field
    def external_bx( F, x, y, z, t, gradient, length_scale ):
        return F + ((z>=0) & (z<=length_scale)) * gradient * y

    def external_by( F, x, y, z, t, gradient, length_scale ):
        return F - ((z>=0) & (z<=length_scale)) * gradient * x

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt,
        p_zmin, p_zmax, p_rmin, p_rmax, p_nz, p_nr, p_nt, n_e,
        dens_func=dens_func, zmin=zmin, initialize_ions=True,
        v_comoving=v_comoving, gamma_boost=gamma_boost, n_order=n_order,
        boundaries={'z':'open', 'r':'open'}, use_cuda=use_cuda, particle_shape='cubic' )

    # Add electron bunch
    z = bunch.xi - np.average(bunch.xi) - L_box/2
    add_particle_bunch_from_arrays(sim, -ct.e, ct.m_e, bunch.x, bunch.y, z,
                                   bunch.px, bunch.py, bunch.pz, w,
                                   boost=boost)

    # Add external fields
    sim.external_fields = [ExternalField(external_bx, 'Bx', g_lens, L_plasma, species=sim.ptcl[2], gamma_boost=gamma_boost),
                           ExternalField(external_by, 'By', g_lens, L_plasma, species=sim.ptcl[2], gamma_boost=gamma_boost)]
    if track_bunch:
        sim.ptcl[2].track( sim.comm )

    # Configure the moving window
    sim.set_moving_window( v=v_window )

    # Add a field diagnostic
    sim.diags = [BoostedFieldDiagnostic( zmin, zmax, ct.c,
                    dt_lab_diag_period, N_lab_diag, gamma_boost,
                    period=write_period, fldobject=sim.fld, comm=sim.comm),
                 BoostedParticleDiagnostic( zmin, zmax, ct.c, dt_lab_diag_period,
                    N_lab_diag, gamma_boost, write_period, sim.fld,
                    select={'uz':[0.,None]}, species={'bunch':sim.ptcl[2]},
                    comm=sim.comm )
                    ]
    # Number of iterations to perform
    N_step = int(T_interact / sim.dt) + write_period
                 
    ### Run the simulation
    sim.step( N_step )
    
    # Analyze last beam
    file_path = 'lab_diags/hdf5/data00000031.h5'
    x, y, z, px, py, pz, q = read_beam('openpmd', file_path, species_name='bunch')
    a_x, b_x, g_x = twiss_parameters(x, px, pz, w=q)

    # Save parameter to file for `analysis_script.py`.
    file_name = 'a_x_abs-{:0.3f}.out'.format(g_lens)
    np.savetxt(file_name, np.array([np.abs(a_x)]))


if __name__ == '__main__':
    g_lens = {{g_lens}}
    model = {{model}}
    run_simulation(g_lens, model)
