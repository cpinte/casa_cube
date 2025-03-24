Examples
========

Basic Analysis of a CO Cube
---------------------------

This example shows how to perform basic analysis on a CO data cube:

.. code-block:: python

   from casa_cube import Cube
   import matplotlib.pyplot as plt

   # Load the data
   cube = Cube('co_cube.fits')

   # Create a figure with multiple panels
   fig, axes = plt.subplots(2, 2, figsize=(12, 12))

   # Plot integrated intensity (M0)
   cube.plot(moment=0, ax=axes[0,0], title='Integrated Intensity')

   # Plot velocity field (M1)
   cube.plot(moment=1, ax=axes[0,1], title='Velocity Field')

   # Plot velocity dispersion (M2)
   cube.plot(moment=2, ax=axes[1,0], title='Velocity Dispersion')

   # Plot peak intensity (M8)
   cube.plot(moment=8, ax=axes[1,1], title='Peak Intensity')

   plt.tight_layout()
   plt.show()

Channel Map Analysis
--------------------

This example demonstrates how to create and analyze channel maps:

.. code-block:: python

   # Create channel maps
   cube.plot_channels(n=20, ncols=5, vmin=-10, vmax=10)

   # Get line profile
   cube.plot_line(x_axis="velocity", threshold=0.1)

   # Calculate and plot moment maps with threshold
   m0 = cube.get_moment_map(moment=0, threshold=0.1)
   m1 = cube.get_moment_map(moment=1, v0=0, threshold=0.1)

   # Plot with custom parameters
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   cube.plot(moment=0, ax=ax1, threshold=0.1,
             cmap='viridis', colorbar=True)
   ax1.set_title('Integrated Intensity')

   cube.plot(moment=1, ax=ax2, v0=0, threshold=0.1,
             cmap='RdBu_r', colorbar=True)
   ax2.set_title('Velocity Field')

   plt.tight_layout()
   plt.show()

High-Pass Filtering
-------------------

This example shows how to apply high-pass filtering to remove large-scale structures:

.. code-block:: python

   # Apply high-pass filter
   filtered_map = cube.get_high_pass_filter_map(
       moment=0,
       w0=5.0,  # 5 arcsec scale
       gamma=0.5  # radial stretch
   )

   # Plot original and filtered maps
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   cube.plot(moment=0, ax=ax1, cmap='viridis')
   ax1.set_title('Original')

   ax2.imshow(filtered_map, cmap='viridis', origin='lower')
   ax2.set_title('High-Pass Filtered')

   plt.tight_layout()
   plt.show()

Advanced Analysis
-----------------

This example demonstrates some advanced features:

.. code-block:: python

   # Convert to brightness temperature
   Tb = cube._Jybeam_to_Tb(cube.get_moment_map(moment=8))

   # Calculate turbulent velocity
   vturb = cube.get_vturb(mol_weight=28)  # for CO

   # Create a cut through the cube
   x, y, z = cube.make_cut(100, 100, 200, 200, num=100)

   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   ax1.imshow(Tb, cmap='viridis', origin='lower')
   ax1.set_title('Brightness Temperature')

   ax2.imshow(vturb, cmap='viridis', origin='lower')
   ax2.set_title('Turbulent Velocity')

   plt.tight_layout()
   plt.show()

   # Plot the cut
   plt.figure(figsize=(8, 6))
   plt.plot(x, z)
   plt.xlabel('Position')
   plt.ylabel('Intensity')
   plt.title('Cut Through Cube')
   plt.show()
