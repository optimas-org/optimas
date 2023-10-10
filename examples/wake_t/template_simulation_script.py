"""Template simulation script for optimizing an APL with Wake-T."""
import numpy as np
from wake_t.utilities.bunch_generation import get_gaussian_bunch_from_twiss
from wake_t.beamline_elements import ActivePlasmaLens
from aptools.data_analysis.beam_diagnostics import twiss_parameters


def run_simulation(g_lens):
    """Run a Wake-T simulation of an active plasma lens."""
    # Generate particle bunch.
    g_x = 600
    b_x = 1
    a_x = -np.sqrt(b_x * g_x - 1)
    em_x = 1e-6  # m
    gamma_avg = 300 / 0.511
    ene_sp = 1  # %
    Q = 100  # pC
    s_t = 10  # fs
    n_part = 1e4
    bunch = get_gaussian_bunch_from_twiss(
        em_x,
        em_x,
        a_x,
        a_x,
        b_x,
        b_x,
        gamma_avg,
        ene_sp,
        s_t,
        0,
        q_tot=Q,
        n_part=n_part,
    )

    # Define plasma lens.
    p_lens = ActivePlasmaLens(3e-2, g_lens, n_out=2)

    # Perform tracking.
    p_lens.track(bunch)

    # Analyze bunch.
    a_x, b_x, g_x = twiss_parameters(bunch.x, bunch.px, bunch.pz, w=bunch.q)

    # Save parameter to file for `analysis_script.py`.
    np.save("a_x_abs", np.abs(a_x))


if __name__ == "__main__":
    g_lens = {{g_lens}}
    run_simulation(g_lens)
