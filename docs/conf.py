# Configuration file for the Sphinx documentation builder

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'IDP-System'
copyright = '2023'
author = 'IDP-System Team'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

autodoc_member_order = 'bysource'
autoclass_content = 'both'
