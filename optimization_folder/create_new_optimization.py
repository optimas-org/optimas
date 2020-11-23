#!/usr/bin/python
import os, shutil, math
import argparse

# Parse the command line
parser = argparse.ArgumentParser(
    description='Prepare a directory, in order to run a new optimization')
parser.add_argument('--name', required=True,
    help='Name of the new directory, where the optimization will be run.')
parser.add_argument('--from_example', required=True,
    choices=sorted(os.listdir('../source/example_sim_specific_folders')),
    help='Example optimization to be copied in the new directory.')
parser.add_argument('--machine', required=True,
    choices=['local'] + os.listdir('../source/submission_scripts'),
    help='Machine on which the simulations will be run.')
parser.add_argument('--n_sim_workers', required=True,
    type=int,
    help='Number of workers that can simultaneously launch simulations')
parser.add_argument('--max_time', default='02:00',
    help='Maximum wall time for the optimization')
args = parser.parse_args()

# TODO: Check / Ask user to provide missing values

# Create corresponding folder, copied from existing example
os.mkdir(args.name)
shutil.copy('../source/run_libensemble.py', args.name)
shutil.copy('../source/simf.py', args.name)
shutil.copytree(
    os.path.join('../source/example_sim_specific_folders', args.from_example),
    os.path.join(args.name, 'sim_specific') )

# Copy relevant submission script
if args.machine != 'local':
    with open(os.path.join('../source/submission_scripts/', args.machine)) as f:
        code = f.read()
    gpu_per_nodes = { 'juwels':4, 'summit':6 }
    n_nodes = int(math.ceil(args.n_sim_workers*1./gpu_per_nodes[args.machine]))
    code = code.format(name=args.name, n_nodes=n_nodes,
                       n_workers=args.n_sim_workers+1, max_time=args.max_time)
    with open( os.path.join(args.name, 'submission_script'), 'w' ) as f:
        f.write(code)

# Print instructions for users
if args.machine=='local':
    command_line = "python run_libensemble.py --comms local --nworkers " + \
        str(args.n_sim_workers + 1)
else:
    submission_command = {'juwels': 'sbatch', 'summit': 'bsub'}
    command_line = submission_command[args.machine] + " submission_script"
message = """
Created a new directory `{name}`.

In order to run the optimization:
--> cd {name}
--> {command_line}
""".format(name=args.name, command_line=command_line)

print(message)

# TODO: Modify README
# TODO: Process sim_max
