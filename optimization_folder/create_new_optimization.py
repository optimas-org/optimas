import os, shutil
import argparse

# Parse the command line
parser = argparse.ArgumentParser()
parser.add_argument('--name', default=None,
    help='Name of the new directory, where the optimization will be run.')
parser.add_argument('--from_example', default=None,
    choices=sorted(os.listdir('example_sim_specific_folders')),
    help='Example optimization to be copied in the new directory.')
args = parser.parse_args()

# TODO: Ask user to provide missing values

# Create corresponding folder, copied from existing example
shutil.copytree(
    os.path.join('example_sim_specific_folders', args.from_example),
    args.name )

# TODO: Print instructions for user

# TODO: Modify README
