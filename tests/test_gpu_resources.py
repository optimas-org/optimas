import os
from multiprocessing import set_start_method

set_start_method("spawn", force=True)

import numpy as np
from optimas.core import VaryingParameter, Objective, Parameter
from optimas.generators import AxSingleFidelityGenerator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration


def analyze_simulation(simulation_directory, output_params):
    """Read in the visible gpus to the worker."""
    # Read back result from file
    with open("cuda_visible_devices.txt") as f:
        cuda_visible_devices = f.read()
    with open("result.txt") as f:
        result = float(f.read())
    # Fill in output parameters.
    output_params["f"] = result
    output_params["cuda_visible_devices"] = cuda_visible_devices
    return output_params


def create_exploration(
    dedicated_resources,
    use_cuda,
    sim_workers,
    gpus_per_worker,
    gpu_id=0,
    exploration_dir_path="./exploration",
):
    # Create varying parameters and objectives.
    var_1 = VaryingParameter("x0", 0.0, 15.0)
    var_2 = VaryingParameter("x1", 0.0, 15.0)
    obj = Objective("f", minimize=True)
    p0 = Parameter("cuda_visible_devices", dtype="U10")

    # Create generator.
    gen = AxSingleFidelityGenerator(
        varying_parameters=[var_1, var_2],
        objectives=[obj],
        analyzed_parameters=[p0],
        n_init=4,
        dedicated_resources=dedicated_resources,
        use_cuda=use_cuda,
        gpu_id=gpu_id,
    )

    # Create evaluator.
    ev = TemplateEvaluator(
        sim_template=os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "resources",
            "template_simulation_script.py",
        ),
        analysis_func=analyze_simulation,
        n_gpus=gpus_per_worker,
    )

    # Create exploration.
    exp = Exploration(
        generator=gen,
        evaluator=ev,
        max_evals=10,
        sim_workers=sim_workers,
        run_async=False,
        exploration_dir_path=exploration_dir_path,
    )

    # Override automatic resource detection to allow testing GPU assignment
    # event if there are no GPUs available.
    exp.libE_specs["resource_info"] = {
        "cores_on_node": (8, 8),
        "gpus_on_node": 4,
    }

    return exp


def test_gpu_workers():
    """Test GPU assignment to simulation workers."""
    exp = create_exploration(
        dedicated_resources=False,
        use_cuda=False,
        sim_workers=4,
        gpus_per_worker=1,
        exploration_dir_path="./tests_output/test_gpu_workers",
    )
    exp.run()
    # Check that all 4 GPUs are used, one per sim worker.
    np.testing.assert_array_equal(
        exp.history["cuda_visible_devices"],
        np.array(exp.history["trial_index"] % 4, dtype=str),
    )


def test_multigpu_workers():
    """
    Test GPU assignment to simulation workers, with multiple GPUs per
    worker.
    """
    exp = create_exploration(
        dedicated_resources=False,
        use_cuda=False,
        sim_workers=2,
        gpus_per_worker=2,
        exploration_dir_path="./tests_output/test_multigpu_workers",
    )
    exp.run()
    # Check that even and odd trials get their corresponding GPUs.
    np.testing.assert_array_equal(
        exp.history.iloc[::2]["cuda_visible_devices"], "0,1"
    )
    np.testing.assert_array_equal(
        exp.history.iloc[1::2]["cuda_visible_devices"], "2,3"
    )


def test_gpu_workers_with_shared_gpu_gen():
    """
    Test GPU assignment to simulation workers, with a gen that runs on a
    shared GPU.
    """
    exp = create_exploration(
        dedicated_resources=False,
        use_cuda=True,
        sim_workers=4,
        gpus_per_worker=1,
        gpu_id=2,
        exploration_dir_path="./tests_output/test_gpu_workers_with_shared_gpu_gen",
    )
    exp.run()
    # Check that gen (manager) gets GPU 2.
    assert os.getenv("CUDA_VISIBLE_DEVICES") == "2"
    # Check all 4 GPUs are available to the workers.
    np.testing.assert_array_equal(
        exp.history["cuda_visible_devices"],
        np.array(exp.history["trial_index"] % 4, dtype=str),
    )


def test_gpu_workers_with_dedicated_gpu_gen():
    """
    Test GPU assignment to simulation workers, with a gen that runs on a
    dedicated GPU.
    """
    exp = create_exploration(
        dedicated_resources=True,
        use_cuda=True,
        sim_workers=3,
        gpus_per_worker=1,
        exploration_dir_path="./tests_output/test_gpu_workers_with_dedicated_gpu_gen",
    )
    exp.run()
    # Check that gen (manager) gets GPU 0.
    assert os.getenv("CUDA_VISIBLE_DEVICES") == "0"
    # Check only 3 GPUs are available to the workers.
    np.testing.assert_array_equal(
        exp.history["cuda_visible_devices"],
        np.array(exp.history["trial_index"] % 3 + 1, dtype=str),
    )


if __name__ == "__main__":
    test_gpu_workers()
    test_multigpu_workers()
    test_gpu_workers_with_shared_gpu_gen()
    test_gpu_workers_with_dedicated_gpu_gen()
