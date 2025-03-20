Installation Guide
=================

Requirements
-----------

Casa Cube requires the following Python packages:

* numpy
* matplotlib
* astropy
* scipy
* cmasher

You can install these dependencies using pip:

.. code-block:: bash

   pip install numpy matplotlib astropy scipy cmasher

Installing Casa Cube
------------------

The easiest way to install Casa Cube is through pip:

.. code-block:: bash

   pip install casa-cube

For development installation:

.. code-block:: bash

   git clone https://github.com/yourusername/casa-cube.git
   cd casa-cube
   pip install -e .

Verifying the Installation
------------------------

You can verify the installation by running Python and importing the package:

.. code-block:: python

   >>> from casa_cube import Cube
   >>> # If no error occurs, the installation was successful

Troubleshooting
--------------

If you encounter any issues during installation:

1. Make sure you have Python 3.7 or higher installed
2. Ensure all required dependencies are installed
3. Check if you have write permissions in your Python environment
4. If using a virtual environment, make sure it's activated before installation

For additional help, please open an issue on the GitHub repository. 