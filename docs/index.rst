Welcome to Casa Cube's documentation!
================================

Casa Cube is a Python package for handling and analyzing radio interferometry data cubes, particularly focused on ALMA data. It provides tools for visualization, analysis, and manipulation of spectral line data cubes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   examples

Installation
-----------

You can install Casa Cube using pip:

.. code-block:: bash

   pip install casa-cube

For development installation:

.. code-block:: bash

   git clone https://github.com/yourusername/casa-cube.git
   cd casa-cube
   pip install -e .

Features
--------

* Read and write FITS data cubes
* Plot channel maps and moment maps
* Calculate various moments (M0, M1, M2, M8, M9)
* Convert between different units (Jy/beam, K)
* Handle beam information
* Support for various coordinate systems
* High-pass filtering capabilities
* Line profile analysis

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 