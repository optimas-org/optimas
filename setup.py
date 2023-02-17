from setuptools import setup, find_packages

from optimas.__version__ import __version__

# Read long description
with open("README.md", "r") as fh:
    long_description = fh.read()


def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f.readlines()]


# Main setup command
setup(name='optimas',
      version=__version__,
      author='',
      author_email="",
      description='Optimization for PIC simulations',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/RemiLehe/fbpic_libE',
      license='',
      packages=find_packages('.'),
      install_requires=read_requirements(),
      platforms='any',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Programming Language :: Python :: 3",
          "Intended Audience :: Science/Research",
          "Operating System :: OS Independent"],
      python_requires=">=3.8",
      ),
