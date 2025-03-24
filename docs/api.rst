API Reference
============

Core Classes
-----------

.. automodule:: casa_cube.Cube
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~Cube.__init__
      ~Cube.plot
      ~Cube.plot_channels
      ~Cube.plot_line
      ~Cube.get_line_profile
      ~Cube.get_moment_map
      ~Cube.get_high_pass_filter_map
      ~Cube.get_fwhm
      ~Cube.get_vturb
      ~Cube.get_std
      ~Cube.make_cut
      ~Cube.explore
      ~Cube.cutout
      ~Cube.tapered_fits

   .. rubric:: Properties

   .. autosummary::
      :nosignatures:

      ~Cube.beam

Constants
--------

.. data:: casa_cube.FWHM_to_sigma
   :annotation: = 1.0 / np.sqrt(8.0 * np.log(2))

.. data:: casa_cube.arcsec
   :annotation: = np.pi / 648000

.. data:: casa_cube.default_cmap
   :annotation: = cmr.arctic

.. data:: casa_cube.line_list
   :annotation: Dictionary of molecular line frequencies and names

Utility Functions
---------------

.. autofunction:: casa_cube.add_colorbar
