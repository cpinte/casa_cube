Usage Guide
===========

Basic Usage
----------

The main class in Casa Cube is the `Cube` class, which handles FITS data cubes. Here's a basic example:

.. code-block:: python

   from casa_cube import Cube

   # Load a FITS cube
   cube = Cube('your_data.fits')

   # Plot a specific channel
   cube.plot(iv=10)  # Plot channel 10

   # Plot a moment map
   cube.plot(moment=0)  # Plot integrated intensity (M0)

   # Plot with custom parameters
   cube.plot(
       moment=1,  # Plot velocity field (M1)
       v0=0,      # Reference velocity
       cmap='RdBu_r',  # Color map
       colorbar=True
   )

Working with Channel Maps
-----------------------

You can plot multiple channels at once using the `plot_channels` method:

.. code-block:: python

   # Plot 20 channels in a 5x4 grid
   cube.plot_channels(n=20, ncols=5)

   # Plot specific velocity range
   cube.plot_channels(vmin=-10, vmax=10)

Calculating Moments
-----------------

Casa Cube provides methods to calculate various moments:

.. code-block:: python

   # Calculate integrated intensity (M0)
   m0 = cube.get_moment_map(moment=0)

   # Calculate velocity field (M1)
   m1 = cube.get_moment_map(moment=1, v0=0)

   # Calculate velocity dispersion (M2)
   m2 = cube.get_moment_map(moment=2)

   # Calculate peak intensity (M8)
   m8 = cube.get_moment_map(moment=8)

   # Calculate peak velocity (M9)
   m9 = cube.get_moment_map(moment=9)

Working with Line Profiles
------------------------

You can analyze line profiles:

.. code-block:: python

   # Get the line profile
   profile = cube.get_line_profile()

   # Plot the line profile
   cube.plot_line(x_axis="velocity")  # Plot vs velocity
   cube.plot_line(x_axis="channel")   # Plot vs channel number
   cube.plot_line(x_axis="freq")      # Plot vs frequency

High-Pass Filtering
-----------------

You can apply high-pass filtering to remove large-scale structures:

.. code-block:: python

   # Apply high-pass filter with 5 arcsec scale
   filtered_map = cube.get_high_pass_filter_map(
       moment=0,  # Apply to integrated intensity
       w0=5.0     # Filter scale in arcsec
   )

Unit Conversions
--------------

Casa Cube can convert between different units:

.. code-block:: python

   # Convert from Jy/beam to brightness temperature
   Tb = cube._Jybeam_to_Tb(flux_map)

   # Convert with Rayleigh-Jeans approximation
   Tb_rj = cube._Jybeam_to_Tb(flux_map, RJ=True)

Working with Beams
----------------

You can access and manipulate beam information:

.. code-block:: python

   # Get beam parameters
   bmaj, bmin, bpa = cube.beam

   # Get beam area
   beam_area = cube._beam_area()  # in arcsec^2
   beam_area_pix = cube._beam_area_pix()  # in pixels^2



Working with scattered light images
----------------

You can specify the pixelscale if it is missing in the header

.. code-block:: python

   obs = casa.Cube('HD169142_2015-05-03_Q_phi.fits',pixelscale=0.01225)
   obs.plot()


Plotting with  RA and Dec on axes instead of relative offsets from image centre
----------------

.. code-block:: python

   obs = casa.Cube('IMLup_continuum.fits')
   ax = plt.subplot(1,1,1,projection=obs.wcs)
   obs.plot(ax)



Advanced Features
---------------

* Tapering: Apply Gaussian tapering to the data
* Dynamic range: Control the dynamic range of plots
* Custom color scales: Use different color scales for different types of data
* Coordinate systems: Work with different coordinate systems (arcsec, au, pixels)

For more examples and advanced usage, see the :doc:`examples` section.
