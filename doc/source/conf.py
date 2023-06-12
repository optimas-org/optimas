# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date
sys.path.insert(0, os.path.abspath('../..'))


# -- Import version ----------------------------------------------------------
from optimas import __version__  # noqa: E402


# -- Project information -----------------------------------------------------
project = 'optimas'
project_copyright = '2023-%s, the optimas collaborators' % date.today().year
author = 'The optimas collaborators'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    # 'sphinx.ext.intersphinx',
    'sphinx_design',
    # 'sphinx_gallery.gen_gallery',
    'numpydoc',
    "matplotlib.sphinxext.plot_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'  # "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Logo
html_logo = "_static/logo.png"
html_favicon = "_static/favicon_128x128.png"

# Theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/optimas-org/optimas",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Slack",
            "url": "https://optimas.slack.com/",
            "icon": "fa-brands fa-slack",
        },
    ],
    "pygment_light_style": "default",
    "pygment_dark_style": "monokai",
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "optimas-org",
    "github_repo": "optimas",
    "github_version": "main",
    "doc_path": "doc/source",
}

# Do not show type hints.
autodoc_typehints = 'none'

# Do  not use numpydoc to generate autosummary.
numpydoc_show_class_members = False

# Create autosummary for all files.
autosummary_generate = True

# Autosummary configuration
autosummary_context = {
    # Methods that should be skipped when generating the docs
    "skipmethods": ["__init__"]
}

# ------------------------------------------------------------------------------
# Matplotlib plot_directive options
# ------------------------------------------------------------------------------

plot_include_source = False
plot_formats = [("png", 300)]
plot_html_show_formats = False
plot_html_show_source_link = False

import math

base_fig_size = 4
phi = (math.sqrt(5) + 1) / 2

font_size = 9

plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.figsize": (base_fig_size * phi, base_fig_size),
    "figure.subplot.bottom": 0.15,
    "figure.subplot.left": 0.15,
    "figure.subplot.right": 0.95,
    "figure.subplot.top": 0.95,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}

# # Configuration for generating tutorials.
# from sphinx_gallery.sorting import FileNameSortKey  # noqa: E402

# sphinx_gallery_conf = {
#      'examples_dirs': '../../tutorials',
#      'gallery_dirs': 'tutorials',
#      'filename_pattern': '.',
#      'within_subsection_order': FileNameSortKey,
# }

# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3/", None),
#     "numpy": ("https://numpy.org/devdocs/", None),
# }
