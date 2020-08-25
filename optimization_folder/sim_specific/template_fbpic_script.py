"""
This is a typical input script that runs a simulation of
laser-wakefield acceleration using FBPIC.

Usage
-----
- Modify the parameters below to suit your needs
- Type "python boosted_frame_script.py" in a terminal

Help
----
All the structures implemented in FBPIC are internally documented.
Enter "print(fbpic_object.__doc__)" to have access to this documentation,
where fbpic_object is any of the objects or function of FBPIC.
"""

# -------
# Imports
# -------
import numpy as np
from scipy.constants import c, e, m_e, m_p
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.laser import add_laser
from fbpic.lpa_utils.bunch import add_particle_bunch_gaussian
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.plasma_mirrors import PlasmaMirror
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, \
        BackTransformedFieldDiagnostic, BackTransformedParticleDiagnostic
# ----------
# Parameters
# ----------
np.random.seed(0)

use_cuda = True

n_order = 32

# Boosted frame
gamma_boost = 60.
# Boosted frame converter
boost = BoostConverter(gamma_boost)

# The simulation box
Nz = 3840        # Number of gridpoints along z
zmax = 0.e-6     # Length of the box along z (meters)
zmin = -161.e-6
Nr = 48          # Number of gridpoints along r
rmax = 242.e-6   # Length of the box along r (meters)
Nm = 3           # Number of modes used

# The simulation timestep
# (See the section Advanced use > Running boosted-frame simulation
# of the FBPIC documentation for an explanation of the calculation of dt)
dt = min( rmax/(2*boost.gamma0*Nr)/c, (zmax-zmin)/Nz/c )  # Timestep (seconds)

# The density profile of the plasma
w_matched = 40.e-6
ramp_up = 2.e-2
plateau = 0.295
ramp_down = 5.e-3
z_start_stages = [0, 0.35, 0.7]

# Parameters of the plasma lenses:
dlen = 0.019
mcce = 510999.
wlen = 0.002
lenses = { 'ga': [ 13950, 25990 ],
           'vb': [299792457.2297312, 299792457.77808934],
           'zlen': [ 0.34, 0.69 ],
           'adjust_factor': [ {{adjust_factor1}}, {{adjust_factor2}}] }

# The lasers (conversion to boosted frame is done inside 'add_laser')
a0 = 1.705685     # Laser amplitude
w0 = 50.e-6       # Laser waist
tau = 7.33841e-14 # Laser duration
lambda0 = 0.8e-6  # Laser wavelength
# Time at which the peak of the laser pulses arrives at the start of each stage
laser_t_peak = [1.46764864e-13, 1.167621098057891e-09, 2.3350954312514234e-09]
# Distance between the focal plane and the entrance of each stage
focal_distance = 0.00875

# The particles of the plasma
p_zmin = 0.e-6   # Position of the beginning of the plasma (meters)
p_zmax = z_start_stages[-1] + plateau + ramp_down
p_rmax = 150.e-6 # Maximal radial position of the plasma (meters)
n_e = 1.7e23     # The density in the labframe (electrons.meters^-3)
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 6         # Number of particles per cell along theta

# Density profile
# Convert parameters to boosted frame
# (NB: the density is converted inside the Simulation object)
ramp_up_b, plateau_b, ramp_down_b = \
    boost.static_length( [ ramp_up, plateau, ramp_down ] )
z_start_stages_b = \
    boost.static_length( z_start_stages )

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
    n = np.zeros_like(z)
    inv_ramp_up_b = 1./ramp_up_b
    inv_ramp_down_b = 1./ramp_down_b
    for z0 in z_start_stages_b:
        zf = z0 + ramp_up_b + plateau_b + ramp_down_b
        # Make ramp up
        n = np.where( (z>=z0) & (z<z0+ramp_up_b),
                      np.sin( 0.5*np.pi*(z-z0)*inv_ramp_up_b )**2, n )
        # Make plateau
        n = np.where( (z>=z0+ramp_up_b) & (z<zf-ramp_down_b),
                      1., n )
        # Make ramp down
        n = np.where( (z>=zf-ramp_down_b) & (z<zf),
                      np.sin( 0.5*np.pi*(zf-z)*inv_ramp_down_b )**2, n )
    # Add transverse guiding parabolic profile
    # Relative change divided by w_matched^2 that allows guiding
    rel_delta_n_over_w2 = 1./( np.pi * 2.81e-15 * w_matched**4 * n_e )
    n = n * ( 1. + rel_delta_n_over_w2 * r**2 )
    return(n)

# The bunch
bunch_sig_r = 6.e-7 # RMS size of the bunch in r
bunch_sig_z = 3.e-6 # RMS size of the bunch is z
bunch_z0 = -109.e-6 # Positions of the bunch at t=0
bunch_n_emit = 2.5e-07 # Normalized emittance
bunch_n_macroparticles = 50000
bunch_n_physical_particles = 1.5e-10/e
bunch_gamma0 = 1957.
bunch_sig_gamma = 39.

# The moving window (moves with the group velocity in a plasma)
v_window = c

# Velocity of the Galilean frame (for suppression of the NCI)
v_comoving =  -0.9998611014647095 * c

# The interaction length of the simulation, in the lab frame (meters)
L_interact = (p_zmax-p_zmin) # the plasma length
# Interaction time, in the boosted frame (seconds)
T_interact = boost.interaction_time( L_interact, (zmax-zmin), v_window )
# (i.e. the time it takes for the moving window to slide across the plasma)

# Number of discrete diagnostic snapshots, for the diagnostics in the lab frame
# (i.e. back-transformed from the simulation frame to the lab frame)
N_lab_diag = 53
# Time interval between diagnostic snapshots *in the lab frame*
# (first at t=0, last at t=T_interact)
dt_lab_diag_period = 0.02/c
# Period of writing the cached, backtransformed lab frame diagnostics to disk
write_period = 50
## Period of the diagnostics in the boosted-frame
diag_period = 1000

# ---------------------------
# Carrying out the simulation
# ---------------------------
# NB: The code below is only executed when running the script,
# (`python boosted_frame_sim.py`), but not when importing it.
if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        v_comoving=v_comoving, gamma_boost=boost.gamma0,
        n_order=n_order, use_cuda=use_cuda,
        particle_shape='cubic', verbose_level=2,
        boundaries={'z':'open', 'r':'reflective'})
        # 'r': 'open' can also be used, but is more computationally expensive

    # Add the plasma electron and plasma ions
    plasma_elec = sim.add_new_species( q=-e, m=m_e,
                    n=n_e, dens_func=dens_func,
                    p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
                    p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )
    plasma_ions = sim.add_new_species( q=e, m=m_p,
                    n=n_e, dens_func=dens_func,
                    p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
                    p_nz=p_nz, p_nr=p_nr, p_nt=p_nt )

    # Add a relativistic electron bunch
    bunch = add_particle_bunch_gaussian( sim, -e, m_e,
                                         sig_r=bunch_sig_r,
                                         sig_z=bunch_sig_z,
                                         n_emit=bunch_n_emit,
                                         gamma0=bunch_gamma0,
                                         sig_gamma=bunch_sig_gamma,
                                         n_physical_particles=bunch_n_physical_particles,
                                         n_macroparticles=bunch_n_macroparticles,
                                         tf=0.0, zf=bunch_z0, boost=boost,
                                         z_injection_plane=0.)
    
    # Add a laser to the fields of the simulation
    for i_stage in range(len(z_start_stages)):
        z0_antenna = z_start_stages[i_stage]
        z0 = z0_antenna - c*laser_t_peak[i_stage]
        zf = z0_antenna + focal_distance
        add_laser( sim, a0, w0, c*tau, z0, lambda0=lambda0,
                   zf=zf, gamma_boost=boost.gamma0,
                   method='antenna', z0_antenna=z0_antenna )

    # Convert parameter to boosted frame
    v_window_boosted, = boost.velocity( [ v_window ] )
    # Configure the moving window
    sim.set_moving_window( v=v_window_boosted )

    # Configure plasma mirrors: at the enf of each stage
    sim.plasma_mirrors = [
        PlasmaMirror( z_lab=z0 + ramp_up+plateau+ramp_down,
                      gamma_boost=gamma_boost, n_cells=4) \
        for z0 in z_start_stages
    ]

    # Configure plasma lenses
    for i_lens in range(len(lenses['ga'])):
        ga = lenses['ga'][i_lens]
        vb = lenses['vb'][i_lens]
        zlen = lenses['zlen'][i_lens]
        adjust_factor = lenses['adjust_factor'][i_lens]

        gab = gamma_boost        
        amplitude = adjust_factor * gab * 4 * mcce * ga / (wlen * dlen)
        
        # Focusing force along x and y
        def Ex( F, x, y, z, t, amplitude, length_scale ):
            return F + amplitude * x * \
                    ((gab*(z+vb*t)>=zlen) & (gab*(z+vb*t)<=zlen+wlen)) 
        def Ey( F, x, y, z, t, amplitude, length_scale ):
            return F + amplitude * y * \
                    ((gab*(z+vb*t)>=zlen) & (gab*(z+vb*t)<=zlen+wlen)) 
        
        sim.external_fields += [
            ExternalField( Ex, 'Ex', amplitude, 0., species=bunch ),
            ExternalField( Ey, 'Ey', amplitude, 0., species=bunch ),
        ]
    
    
    # Add a field diagnostic
    sim.diags = [
                  # Diagnostics in the boosted frame
                  FieldDiagnostic( diag_period,
                        fldobject=sim.fld, comm=sim.comm ),
                  ParticleDiagnostic( diag_period,
                        species={"bunch":bunch}, comm=sim.comm),
                  # Diagnostics in the lab frame (back-transformed)
                  BackTransformedFieldDiagnostic( zmin, zmax, v_window,
                                                  dt_lab_diag_period,
                                                  N_lab_diag, boost.gamma0,
                                                  fieldtypes=['rho','E'],
                                                  period=write_period,
                                                  fldobject=sim.fld,
                                                  comm=sim.comm ),
                  BackTransformedParticleDiagnostic( zmin, zmax, v_window,
                                                  dt_lab_diag_period,
                                                  N_lab_diag, boost.gamma0,
                                                  write_period, sim.fld, 
                                                  species={'bunch':bunch},
                                                  comm=sim.comm )
                ]

    # Number of iterations to perform
    N_step = int(T_interact/sim.dt)

    ### Run the simulation
    sim.step( N_step )
    print('')
