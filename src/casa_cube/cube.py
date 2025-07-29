import os
from copy import deepcopy

import cmasher as cmr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from scipy import ndimage
from warnings import catch_warnings, simplefilter

FWHM_to_sigma = 1.0 / np.sqrt(8.0 * np.log(2)) # 1/2.355  FWHM = sigma * sqrt(8ln(2))
arcsec = np.pi / 648000
default_cmap = cmr.arctic


class Cube:
    """
    A class to handle astronomy data cubes. This class provides functionality to read, manipulate,
    and analyze 3D FITS data cubes commonly used in (radio) astronomy.

    The cube data is expected to have 3 dimensions:
    - Two spatial dimensions (RA, Dec)
    - One spectral dimension (frequency or velocity)

    The class provides methods for:
    - Reading FITS files
    - Basic cube manipulation (cropping, smoothing, etc.)
    - Moment map generation
    - Spectral analysis
    - Visualization

    Attributes:
        filename (str): Path to the input FITS file
        header (astropy.io.fits.Header): FITS header information
        data (numpy.ndarray): 3D array containing the cube data
        wcs (astropy.wcs.WCS): World Coordinate System information
        object (str): Name of the astronomical object
        unit (str): Units of the data values

    """

    def __init__(self, filename, only_header=False, correct_factor=None, unit=None, pixelscale=None, restfreq=None, zoom=None, **kwargs):
        """
        Initialize a Cube object.

        Args:
            filename (str): The path to the FITS file containing the cube data.
            only_header (bool): If True, only read the header and not the data.
        """

        self.filename = os.path.normpath(os.path.expanduser(filename))
        self._read(**kwargs, only_header=only_header, correct_factor=correct_factor, unit=unit, pixelscale=pixelscale, restfreq=restfreq, zoom=zoom)


    def _read(self, only_header=False, correct_factor=None, unit=None, pixelscale=None, instrument=None, restfreq=None, zoom=None):
        """
        Read the FITS file and initialize the Cube object.

        Args:
            only_header (bool): If True, only read the header and not the data.
            correct_factor (array-like): Array of correction factors for the data.
            unit (str): The unit of the data.
            pixelscale (float): The pixel scale in arcsec.
            instrument (str): The instrument used to acquire the data.
            restfreq (float): The rest frequency of the data in Hz.
            zoom (float): The zoom factor for the data.
        """

        try:
            hdu = fits.open(self.filename)
            self.header = hdu[0].header

            # Read a few keywords in header
            try:
                self.object = hdu[0].header['OBJECT']
            except:
                print("Warning: could not find header keyword OBJECT")
                self.object = ""

            try:
                self.unit = hdu[0].header['BUNIT']
            except:
                print("Warning: could not find header keyword BUNIT")
                self.unit = ""

            if unit is not None:
                print("Warning: forcing unit")
                self.unit=unit

            if self.unit == "beam-1 Jy": # discminer format
                self.unit = "Jy/beam"

            if self.unit == "JY/PIXEL": # radmc format
                self.unit = "Jy pixel-1"

            # Suppress warnings when creating WCS
            with catch_warnings():
                simplefilter("ignore")  # Ignore all warnings
                try:
                    self.wcs = WCS(self.header).celestial
                except Exception as e:
                    print(f"Warning: could not create WCS - {e}")
                    self.wcs = None

            # pixel info
            self.nx = hdu[0].header['NAXIS1']
            self.ny = hdu[0].header['NAXIS2']
            try:
                try:
                   self.pixelscale = hdu[0].header['CDELT2'] * 3600 # arcsec
                except KeyError:
                   self.pixelscale = abs(hdu[0].header['CD2_2']) * 3600
                self.cx = hdu[0].header['CRPIX1']
                self.cy = hdu[0].header['CRPIX2']
                self.x_ref = hdu[0].header['CRVAL1']  # coordinate
                self.y_ref = hdu[0].header['CRVAL2']
            except:
                print("Warning: missing WCS")
                self.cx = self.nx//2 + 1
                self.cy = self.ny//2 + 1
                self.x_ref = 0
                self.y_ref = 0

                # SPHERE pixelscales (Maire et al 2016)
                if instrument == "IFS":
                    pixelscale = 7.46e-3
                elif instrument == "IRDIS_Y2":
                    pixelscale = 12.283e-3
                elif instrument == "IRDIS_Y3":
                    pixelscale = 12.283e-3
                elif instrument == "IRDIS_J2":
                    pixelscale = 12.266e-3
                elif instrument == "IRDIS_J3":
                    pixelscale = 12.261e-3
                elif instrument == "IRDIS_H2":
                    pixelscale = 12.255e-3
                elif instrument == "IRDIS_H3":
                    pixelscale = 12.250e-3
                elif instrument == "IRDIS_K1":
                    pixelscale = 12.267e-3
                elif instrument == "IRDIS_K2":
                    pixelscale = 12.263e-3
                elif instrument == "IRDIS_BB_J":
                    pixelscale = 12.263e-3
                elif instrument == "IRDIS_BB_H":
                    pixelscale = 12.251e-3
                elif instrument == "IRDIS_BB_Ks":
                    pixelscale = 12.265e-3

                if pixelscale is None:
                    raise ValueError("Please provide pixelscale (or instrument)")
                self.pixelscale = pixelscale
                print(f"Pixel scale set to {self.pixelscale} mas")

            self.FOV = np.maximum(self.nx, self.ny) * self.pixelscale

            # image axes : with 0, 0 assumed as the center of the image
            # (Need to add self.x_ref or y_ref for full coordinates)
            self.xaxis = -(np.arange(1, self.nx + 1) - self.cx) * self.pixelscale
            self.yaxis = (np.arange(1, self.ny + 1) - self.cy) * self.pixelscale

            # velocity axis
            try:
                self.nv = hdu[0].header['NAXIS3']
            except:
                self.nv = 1
            try:
                self.restfreq = hdu[0].header['RESTFRQ']
                self.wl = sc.c / self.restfreq
            except:
                try:
                    self.restfreq = hdu[0].header['RESTFREQ']  # gildas format
                    self.wl = sc.c / self.restfreq
                except:
                    if restfreq is None:
                        print("Warning: missing rest frequency")
                    else:
                        self.restfreq = restfreq
                        self.wl = sc.c / self.restfreq

            try:
                self.velocity_type = hdu[0].header['CTYPE3']
                self.CRPIX3 = hdu[0].header['CRPIX3']
                self.CRVAL3 = hdu[0].header['CRVAL3']
                self.CDELT3 = hdu[0].header['CDELT3']
                if self.velocity_type == "VELO-LSR" or  self.velocity_type == "VRAD": # gildas and casa
                    try:
                        self.CUNIT3 = hdu[0].header['CUNIT3']
                    except:
                        self.CUNIT3 = None
                    if self.CUNIT3 == "M/S":
                        factor = 1e-3
                    elif self.CUNIT3 == "KM/S":
                        factor = 1
                    else:
                        if self.CDELT3 < 5: # assuming km/s
                            print("Assuming velocity axis is in km/s")
                            factor = 1
                        else: # assuming m/s
                            factor = 1e-3
                            print("Assuming velocity axis is in m/s")
                    self.velocity = (self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)) * factor # km/s
                    self.nu = self.restfreq * (1 - self.velocity * 1000 / sc.c)
                elif self.velocity_type == "FREQ" or self.velocity_type=="FREQ-LSR": # Hz
                    self.nu = self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)
                    self.velocity = (-(self.nu - self.restfreq) / self.restfreq * sc.c / 1000.0)  # km/s
                else:
                    raise ValueError("Velocity type is not recognised:", self.velocity_type)
                self.is_V = True
            except:
                print("Warning: could not extract velocity")
                self.is_V = False

            # beam
            try:
                self.bmaj = hdu[0].header['BMAJ'] * 3600 # arcsec
                self.bmin = hdu[0].header['BMIN'] * 3600
                self.bpa = hdu[0].header['BPA']
            except:
                try:
                    # make an average of all the records ...
                    self.bmaj = hdu[1].data[0][0]
                    self.bmin = hdu[1].data[0][1]
                    self.bpa = hdu[1].data[0][2]
                except:
                    print("Warning: missing beam")
                    self.bmaj = 0
                    self.bmin = 0
                    self.bpa = 0

            # reading data
            if not only_header:
                self.image = np.ma.masked_array(hdu[0].data)

                if self.image.ndim == 4:
                    self.image = self.image[0, :, :, :]

                if self.image.ndim == 3 and self.nv == 1:
                    self.image = self.image[0, :, :]

                if correct_factor is not None:
                    self.image *= correct_factor[:,np.newaxis, np.newaxis]

                if zoom is not None:
                    print('Original size = ', self.image.shape)
                    self.image[np.isnan(self.image)] = 0.
                    self.image = ndimage.zoom(self.image, zoom=[1, zoom, zoom])
                    print('Resampled size = ', self.image.shape)
                    nx = self.image.shape[-1]
                    ny = self.image.shape[-2]
                    self.pixelscale = self.pixelscale*self.nx/nx
                    self.nx = nx
                    self.ny = ny

            hdu.close()
        except OSError:
            print('cannot open', self.filename)
            return ValueError

    def cutout(self, filename, FOV=None, ix_min=None, ix_max=None, iv_min=None, iv_max=None, vmin=None, vmax=None, no_pola=False, pmin=None, pmax=None, channels=None, **kwargs):
        """
        Cut out a region (in space, frequency and polarization) of the cube and save it as a new FITS file.

        Args:
            filename (str): The path to the output FITS file.
            FOV (float): The field of view in arcsec.
            ix_min (int): The minimum index of the x-axis.
            ix_max (int): The maximum index of the x-axis.
            iv_min (int): The minimum index of the velocity axis.
            iv_max (int): The maximum index of the velocity axis.
            vmin (float): The minimum velocity in km/s.
            vmax (float): The maximum velocity in km/s.
            no_pola (bool): If True, only keep the first polarization.
            pmin (int): The minimum index of the polarization axis.
            pmax (int): The maximum index of the polarization axis.
            channels (list): The list of channels to keep.
            **kwargs: Additional keyword arguments for the fits.writeto function.
        """

        image = deepcopy(self.image)
        header = deepcopy(self.header)

        ndim = image.ndim

        # ---- Spatial trimming
        if FOV is not None:
            cutout_pix = int(FOV / self.pixelscale)-1
            while cutout_pix * self.pixelscale < FOV:
                cutout_pix += 1
                excess_pix = int(0.5 * (self.nx - cutout_pix))

                ix_min=excess_pix
                ix_max=excess_pix+cutout_pix

        # We do not trim by default
        if ix_min is None:
            ix_min=0
        if ix_max is None:
            ix_max=self.nx

        # forcing square image for now
        iy_min = ix_min
        iy_max = ix_max

        # Updating header
        if header['CRPIX1']-1 >= ix_min: # we keep the same pixel, and adjust its index
            header['CRPIX1'] -= ix_min
        else: # we use the first pixel, and update its coordinates
            header['CRVAL1'] += (ix_min - header['CRPIX1']-1) * header['CDELT1']
            header['CRPIX1'] = 1
        header['CRPIX2'] = ix_max-ix_min

        if header['CRPIX2']-1 >= iy_min: # we keep the same pixel, and adjust its index
            header['CRPIX2'] -= iy_min
        else: # we use the first pixel, and update its coordinates
            header['CRVAL2'] += (iy_min - header['CRPIX2']-1) * header['CDELT2']
        header['CRPIX2'] = iy_max-iy_min

        # ---- Spectral trimming
        if vmin is not None:
            iv_min = np.argmin(np.abs(self.velocity - vmin))
        if vmax is not None:
            iv_max = np.argmin(np.abs(self.velocity - vmax))

        # We do not trim by default
        if iv_min is None:
            iv_min=0
        if iv_max is None:
            iv_max=self.nv

        # if the velocity axis is flipped
        if iv_min > iv_max:
            temp = iv_min
            iv_min = iv_max
            iv_max = temp

        # Updating header
        if header['CRPIX3']-1 >= iv_min: # we keep the same pixel, and adjust its index
            header['CRPIX3'] -= iv_min
        else: # we use the first pixel, and update its coordinates
            header['CRVAL3'] += (iv_min - (header['CRPIX3']-1)) * header['CDELT3']
            header['CRPIX3'] = 1
        header['NAXIS3'] = iv_max - iv_min

        # --- Polarisation trimming
        if ndim > 3:
            if no_pola:
                pmin = 0
                pmax = 0
                header['NAXIS4'] = 1

        # trimming cube
        if image.ndim == 4:
            image = image[pmin:pmax,iv_min:iv_max,iy_min:iy_max,ix_min:ix_max]
        elif image.ndim == 3:
            image = image[iv_min:iv_max,iy_min:iy_max,ix_min:ix_max]
        else:
            raise ValueError("incorrect dimension in fits file")

        fits.writeto(os.path.normpath(os.path.expanduser(filename)),image.data, header, **kwargs)

        return


    def tapered_fits(self, filename, taper=None, **kwargs):
        """
        Create a tapered version of a fits file.

        Args:
            filename (str): The path to the output FITS file.
            taper (float): The taper size in arcsec.
            **kwargs: Additional keyword arguments for the fits.writeto function.
        """

        if taper is None:
            raise ValueError("taper is needed")

        image = deepcopy(self.image)
        header = deepcopy(self.header)

        ndim = image.ndim
        nv=image.shape[0]

        if taper < self.bmaj:
            print("taper is smaller than bmaj=", self.bmaj)
            print("No taper applied")
        else:
            delta_bmaj = np.sqrt(taper ** 2 - self.bmaj ** 2)
            bmaj = taper

            delta_bmin = np.sqrt(taper ** 2 - self.bmin ** 2)
            bmin = taper

            sigma_x = delta_bmin / self.pixelscale * FWHM_to_sigma  # in pixels
            sigma_y = delta_bmaj / self.pixelscale * FWHM_to_sigma  # in pixels

            print("beam = ", self.bmaj, self.bmin)
            print("tapper =", delta_bmaj, delta_bmin, self.bpa)
            print("beam = ",np.sqrt(self.bmaj ** 2 + delta_bmaj ** 2),np.sqrt(self.bmin ** 2 + delta_bmin ** 2))

            beam = Gaussian2DKernel(sigma_x, sigma_y, self.bpa * np.pi / 180)

            for iv in range(nv):
                im = deepcopy(self.image[iv,:,:])
                im = convolve_fft(im, beam)

                # Correcting for beam area. Tested ok
                im *= taper**2/(self.bmin * self.bmaj)

                image[iv,:,:] = im

            header['BMAJ'] = taper/3600
            header['BMIN'] = taper/3600

            fits.writeto(os.path.normpath(os.path.expanduser(filename)),image.data, header, **kwargs, overwrite=True)

        return


    def sensitivity(self):
        """
        Calculate the sensitivity of the cube (in K).
        """

        self.get_std()
        T = self._Jybeam_to_Tb(self.std,RJ=True)

        return T


    def writeto(filename, image, header, **kwargs):
        """
        Write the cube to a FITS file.
        """

        fits.writeto(os.path.normpath(os.path.expanduser(filename)),image.data, header, **kwargs)

    def plot(
        self,
        iv=None,
        v=None,
        colorbar=True,
        plot_beam=True,
        color_scale=None,
        fmin=None,
        fmax=None,
        limit=None,
        limits=None,
        moment=None,
        moment_fname=None,
        vturb = False,
        Tb=False,
        cmap=None,
        v0=None,
        dv=None,
        ax=None,
        no_ylabel=False,
        no_xlabel=False,
        no_vlabel=False,
        title=None,
        alpha=1.0,
        interpolation="bicubic",
        resample=0,
        bmaj=None,
        bmin=None,
        bpa=None,
        taper=None,
        colorbar_label=True,
        colorbar_side="right",
        M0_threshold=None,
        M8_threshold=None,
        threshold = None,
        threshold_value=np.nan,
        vlabel_position="bottom",
        vlabel_color="white",
        vlabel_size=8,
        shift_dx=0,
        shift_dy=0,
        mol_weight=None,
        iv_support=None,
        v_minmax = None,
        axes_unit = "arcsec",
        quantity_name=None,
        stellar_mask = None,
        levels=4,
        plot_type="imshow",
        linewidths=None,
        zorder=None,
        per_arcsec2=False,
        colors=None,
        x_beam = 0.125,
        y_beam = 0.125,
        mJy=False,
        width=None,
        highpass_filter=0,
        normalise=False,
        dynamic_range=None,
        hpf=False,
        **kwargs
    ):
        """
        Plotting routine for continuum image, moment maps and channel maps.

        Args:
            iv (int): The index of the velocity channel to plot.
            v (float): The velocity in km/s.
            colorbar (bool): Whether to plot the colorbar.
            plot_beam (bool): Whether to plot the beam.
            color_scale (str): The color scale to use.
            fmin (float): The minimum value of the color scale.
            fmax (float): The maximum value of the color scale.
            limit (float): The limit of the color scale.
            limits (list): The limits of the color scale.
            moment (int): The moment to plot.
            moment_fname (str): The filename of the moment map.
            vturb (bool): Whether to plot the turbulent velocity.
            Tb (bool): Whether to plot the brightness temperature.
            cmap (str): The colormap to use.
            v0 (float): The central velocity.
            dv (float): The velocity width.
            ax (matplotlib.axes.Axes): The axes to plot on.
            no_ylabel (bool): Whether to hide the y-axis label.
            no_xlabel (bool): Whether to hide the x-axis label.
            no_vlabel (bool): Whether to hide the velocity label.
            title (str): The title of the plot.
            alpha (float): The alpha value of the plot.
            interpolation (str): The interpolation method to use.
            resample (int): The resampling factor.
            bmaj (float): The major axis of the beam.
            bmin (float): The minor axis of the beam.
            bpa (float): The position angle of the beam.
            taper (float): The taper size.
            colorbar_label (bool): Whether to show the colorbar label.
            colorbar_side (str): The side of the colorbar to plot.
            M0_threshold (float): The threshold for the M0 moment map.
            M8_threshold (float): The threshold for the M8 moment map.
            threshold (float): The threshold for the plot.
            threshold_value (float): The value to use for the threshold.
            vlabel_position (str): The position of the velocity label.
            vlabel_color (str): The color of the velocity label.
            vlabel_size (int): The size of the velocity label.
            shift_dx (float): The shift in the x-direction.
            shift_dy (float): The shift in the y-direction.
            mol_weight (str): The molecular weight to use.
            iv_support (list): The indices of the velocity channels to support.
            v_minmax (list): The minimum and maximum velocities to plot.
            axes_unit (str): The unit of the axes.
            quantity_name (str): The name of the quantity to plot.
            stellar_mask (str): The filename of the stellar mask.
            levels (int): The number of levels for the contour plot.
            plot_type (str): The type of plot to make.
            linewidths (list): The linewidths for the contour plot.
            zorder (int): The zorder for the plot.
            per_arcsec2 (bool): Whether to plot the quantity per arcsec^2.
            colors (list): The colors for the contour plot.
            x_beam (float): The x-axis beam size.
            y_beam (float): The y-axis beam size.
            mJy (bool): Whether to plot the quantity in mJy.
            width (float): The width of the plot.
            highpass_filter (float): The highpass filter to apply.
            normalise (bool): Whether to normalise the plot.
            dynamic_range (float): The dynamic range of the plot.
            hpf (bool): Whether to apply a highpass filter.
            **kwargs: Additional keyword arguments for the plot.
        """


        if ax is None:
            ax = plt.gca()

        # Automatically check if the plot has been opened with projection='wcs'
        use_wcs = hasattr(ax, 'coords') and ax.coords is not None

        unit = self.unit

        if self.nv == 1:  # continuum image
            is_cont = True
            if self.image.ndim > 2:
                im = self.image[0, :, :]
            else:
                im = self.image
            _color_scale = 'log'
        elif moment is not None:
            is_cont = False
            if moment_fname is not None:
                hdu = fits.open(moment_fname)
                im = hdu[0].data
            else:
                if hpf:
                    im = self.get_high_pass_filter_map(moment=moment, v0=v0, M0_threshold=M0_threshold, M8_threshold=M8_threshold, threshold=threshold, iv_support=iv_support, v_minmax=v_minmax, **kwargs)
                else:
                    im = self.get_moment_map(moment=moment, v0=v0, M0_threshold=M0_threshold, M8_threshold=M8_threshold, threshold=threshold, iv_support=iv_support, v_minmax=v_minmax)
            _color_scale = 'lin'
        elif vturb:
            is_cont = False
            im = self.get_vturb(M0_threshold=M0_threshold, threshold=threshold, mol_weight=mol_weight)
            _color_scale = 'lin'

        else:
            if self.is_V:
                is_cont = False
                # -- Selecting channel corresponding to a given velocity
                if dv is not None:
                    v = v0 + dv

                if v is not None:
                    iv = np.abs(self.velocity - v).argmin()
                    print("Selecting channel #", iv)

                if iv is None:
                    print("Channel or velocity needed")
                    return ValueError
            else:
                is_cont = True

            # Averaging multiple channels
            v_offset = 0.0
            try:
                dv = np.diff(self.velocity)[0]
            except:
                dv = None

            if width is not None:
                n_channels = np.maximum(int(np.round(width/dv)),1)

                if n_channels%2: # odd number of channels, same central channel
                    delta_iv = n_channels//2
                    iv_min=iv - delta_iv
                    iv_max=iv + delta_iv+1
                else: # We will have a small shift compared to initial channel
                    delta_iv = n_channels//2
                    iv_min=iv - delta_iv
                    iv_max=iv + delta_iv
                    v_offset = -0.5*dv

                print("Averaging between channels", iv_min, "and", iv_max-1, "(included). Width is", self.velocity[iv_max]-self.velocity[iv_min],"km/s")

                im = np.average(self.image[iv_min:iv_max, :, :],axis=0)

            else: # 1 single channel
                im = self.image[iv, :, :]
            _color_scale = 'lin'


        # --- Convolution by taper
        if taper is not None:
            if taper < self.bmaj:
                print("taper is smaller than bmaj=", self.bmaj)
                print("No taper applied")
            else:
                delta_bmaj = np.sqrt(taper ** 2 - self.bmaj ** 2)
                bmaj = taper

                delta_bmin = np.sqrt(taper ** 2 - self.bmin ** 2)
                bmin = taper

                sigma_x = delta_bmin / self.pixelscale * FWHM_to_sigma  # in pixels
                sigma_y = delta_bmaj / self.pixelscale * FWHM_to_sigma  # in pixels

                print("beam = ", self.bmaj, self.bmin)
                print("tapper =", delta_bmaj, delta_bmin, self.bpa)
                print("beam = ",np.sqrt(self.bmaj ** 2 + delta_bmaj ** 2),np.sqrt(self.bmin ** 2 + delta_bmin ** 2))

                beam = Gaussian2DKernel(sigma_x, sigma_y, self.bpa * np.pi / 180)
                im = convolve_fft(im, beam)

                # Correcting for beam area
                im *= taper**2/(self.bmin * self.bmaj)


        if Tb:
            im = self._Jybeam_to_Tb(im)
            unit = "K"
            #if unit == "Jy/beam":
            #    im = self._Jybeam_to_Tb(im)
            #    unit = "K"
            #else:
            #    print("Unknown unit, don't know kow to convert to Tb")
            #    return ValueError
            _color_scale = 'lin'

        if mJy:
            im = im*1e3
            unit = unit.replace("Jy","mJy")

        if per_arcsec2:
            im = im / self.pixelscale**2
            unit = unit.replace("pixel-1", "arcsec-2")

        if normalise:
            im = im/np.max(im)

        # --- resampling
        if resample > 0:
            mask = ndimage.zoom(im.mask * 1, resample, order=3)
            im = ndimage.zoom(im.data, resample, order=3)
            im = np.ma.masked_where(mask > 0.0, im)

        # -- default color scale
        if color_scale is None:
            color_scale = _color_scale

        # --- Cuts
        if fmax is None:
            fmax = np.nanmax(im)
        if fmin is None:
            if color_scale == 'log':
                fmin = fmax * 1e-2
            else:
                fmin = 0.0

        if dynamic_range is not None:
            fmin = fmax * dynamic_range


        # -- set up the color scale
        print("fminmax=", fmin, fmax)

        if color_scale == 'log':
            norm = mcolors.LogNorm(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'lin':
            norm = mcolors.Normalize(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'sqrt':
            norm = mcolors.PowerNorm(0.5, vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'asinh':
            norm = mcolors.AsinhNorm(vmin=0.5*fmax, vmax=fmax, clip=True)
        else:
            raise ValueError("Unknown color scale: " + color_scale)

        if cmap is None:
            if moment in [1, 9]:
                cmap = "RdBu_r"
            else:
                cmap = default_cmap

        if colors is not None:
            cmap = None


        # option to use WCS coordinates
        if use_wcs:
            xlabel = 'RA (J2000)'
            ylabel = 'Dec (J2000)'
            ax.coords[0].set_axislabel(xlabel)
            ax.coords[1].set_axislabel(ylabel)
            ax.set_frame_on(False)
            extent = None
        else:
            if axes_unit.lower() == 'arcsec':
                pix_scale = self.pixelscale
                xlabel = r'$\Delta$ RA (")'
                ylabel = r'$\Delta$ Dec (")'
                xaxis_factor = -1
            elif axes_unit.lower() == 'au':
                pix_scale = self.pixelscale * self.P.map.distance
                xlabel = 'Distance from star (au)'
                ylabel = 'Distance from star (au)'
                xaxis_factor = 1
            elif axes_unit.lower() == 'pixels' or axes_unit.lower() == 'pixel':
                pix_scale = 1
                xlabel = r'x (pix)'
                ylabel = r'y (pix)'
                xaxis_factor = 1
            else:
                raise ValueError("Unknown unit for axes_units: " + axes_unit)

            halfsize = np.asarray(im.shape) / 2 * pix_scale

            extent = [-halfsize[1]*xaxis_factor-shift_dx, halfsize[1]*xaxis_factor-shift_dx, -halfsize[0]-shift_dy, halfsize[0]-shift_dy]
            if axes_unit.lower() == 'pixels' or axes_unit.lower() == 'pixel':
                extent = None

        self.extent = extent

        if threshold is not None:
            im = np.where(im > threshold, im, threshold_value)

        if highpass_filter > 0:
            sigma =  self.bmaj / self.pixelscale * FWHM_to_sigma * highpass_filter
            kernel =  Gaussian2DKernel(sigma, sigma, 0)
            im -= convolve_fft(im, kernel)

        if plot_type=="imshow":
            image = ax.imshow(
                im,
                norm=norm,
                extent=extent,
                origin='lower',
                cmap=cmap,
                alpha=alpha,
                interpolation=interpolation,
                zorder=zorder
            )
        elif plot_type=="contourf":
            image = ax.contourf(
                im,
                extent=extent,
                origin='lower',
                levels=levels,
                cmap=cmap,
                linewidths=linewidths,
                alpha=alpha,
                zorder=zorder
            )
        elif plot_type=="contour":
            imagee = ax.contour(
                im,
                extent=extent,
                origin='lower',
                levels=levels,
                cmap=cmap,
                linewidths=linewidths,
                alpha=alpha,
                zorder=zorder,
                colors=colors
            )

        if limit is not None:
            limits = [limit, -limit, -limit, limit]

        if limits is not None:
            if use_wcs:
                wcs = WCS(self.header).celestial
                # Convert world coordinates (RA/Dec) to pixel coordinates
                pixels = wcs.wcs_world2pix(
                    [[limits[0], limits[2]], [limits[1], limits[3]]],
                    0
                )
                x1, y1 = pixels[0]  # First point (ra_max, dec_min)
                x2, y2 = pixels[1]  # Second point (ra_min, dec_max)
                ax.set_xlim(x1, x2)
                ax.set_ylim(y1, y2)
            else:
                ax.set_xlim(limits[0], limits[1])
                ax.set_ylim(limits[2], limits[3])

        if not no_xlabel:
            ax.set_xlabel(xlabel)
        if not no_ylabel:
            ax.set_ylabel(ylabel)

        if title is not None:
            axe.set_title(title)

        # -- Color bar
        if colorbar:
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            #cb = plt.colorbar(image, cax=cax, extend=colorbar_extend)
            cb = add_colorbar(image,side=colorbar_side)

            # cax,kw = mpl.colorbar.make_axes(ax)
            # cb = plt.colorbar(image,cax=cax, **kw)
            formatted_unit = unit.replace("-1", "$^{-1}$").replace("-2", "$^{-2}$")

            if colorbar_label:
                if moment == 0:
                    cb.set_label("Flux (" + formatted_unit + "$\,$km$\,$s$^{-1}$)")
                elif moment in [1, 9]:
                    cb.set_label("Velocity (km$\,$s$^{-1})$")
                elif moment == 2:
                    cb.set_label("Velocity dispersion (km$\,$s$^{-1}$)")
                else:
                    if Tb:
                        cb.set_label("T$_\mathrm{B}$ (" + formatted_unit + ")")
                    else:
                        if quantity_name is None:
                            quantity_name = "Flux"
                        if len(formatted_unit) > 0:
                            formatted_unit = " (" + formatted_unit + ")"
                        cb.set_label(quantity_name+formatted_unit)
            plt.sca(ax)  # we reset the main axis

        # -- Adding velocity
        if vlabel_position == "top":
            y_vlabel = 0.85
            x_vlabel = 0.5
        elif vlabel_position == "top-right":
            y_vlabel = 0.85
            x_vlabel = 0.7
        elif vlabel_position == "top-left":
            y_vlabel = 0.85
            x_vlabel = 0.25
        else:
            y_vlabel = 0.1
            x_vlabel = 0.5
        if not no_vlabel:
            if (moment is None) and not is_cont and not vturb:
                if v0 is None:
                    ax.text(
                        x_vlabel,
                        y_vlabel,
                        f"v={self.velocity[iv]+v_offset:<4.2f}$\,$km/s",
                        horizontalalignment='center',
                        color=vlabel_color,
                        transform=ax.transAxes,
                        fontsize=vlabel_size
                    )
                else:
                    ax.text(
                        x_vlabel,
                        y_vlabel,
                        f"$\Delta$v={self.velocity[iv] -v0:<4.2f}$\,$km/s",
                        horizontalalignment='center',
                        color="white",
                        transform=ax.transAxes,
                        fontsize=vlabel_size
                    )

        # --- Adding beam
        if plot_beam:
            # In case the beam is wrong in the header, when can pass the correct one
            if bmaj is None:
                bmaj = self.bmaj
            if bmin is None:
                bmin = self.bmin
            if bpa is None:
                bpa = self.bpa

            beam = Ellipse(
                ax.transLimits.inverted().transform((x_beam, y_beam)),
                width=bmin,
                height=bmaj,
                angle=-bpa,
                fill=True,
                color="grey",
            )
            ax.add_patch(beam)

        # Adding mask to hide star
        if stellar_mask is not None:
            dx = 0.5
            dy = 0.5
            mask = Ellipse(
                ax.transLimits.inverted().transform((dx, dy)),
                width=2 * stellar_mask,
                height=2 * stellar_mask,
                fill=True,
                color='grey',
            )
            ax.add_patch(mask)

        #-- Saving the last plotted quantity
        self.last_image = im

        return image

    def plot_channels(self,n=20, num=21, ncols=5, iv_min=None, iv_max=None, vmin=None, vmax=None, **kwargs):
        """
        Plot the channels of the cube.

        Args:
            n (int): The number of channels to plot.
            num (int): The figure number.
            ncols (int): The number of columns in the plot.
            iv_min (int): The minimum index of the velocity channel to plot.
            iv_max (int): The maximum index of the velocity channel to plot.
            vmin (float): The minimum velocity to plot.
            vmax (float): The maximum velocity to plot.
            **kwargs: Additional keyword arguments for the plot.
        """

        if vmin is not None:
            iv_min = np.abs(self.velocity - vmin).argmin()

        if vmax is not None:
            iv_max = np.abs(self.velocity - vmax).argmin()

        if iv_min is None:
            iv_min = 0
        if iv_max is None:
            iv_max = self.nv-1

        nv = iv_max-iv_min
        dv = nv/n

        nrows = np.ceil(n / ncols).astype(int)

        if (plt.fignum_exists(num)):
            plt.figure(num)
            plt.clf()
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(11, 2*nrows+1),num=num,clear=False)

        for i, ax in enumerate(axs.flatten()):

            if (i%ncols ==0):
                no_ylabel=False
            else:
                no_ylabel=True

            if (i>=ncols*(nrows-1)):
                no_xlabel = False
            else:
                no_xlabel = True

            self.plot(iv=int(iv_min+i*dv), ax=ax, no_xlabel=no_xlabel, no_ylabel=no_ylabel, **kwargs)

        plt.show()

        return


    def get_line_profile(self,threshold=None, **kwargs):
        """
        Get the line profile of the cube.

        Args:
            threshold (float): The threshold for the line profile.
            **kwargs: Additional keyword arguments for the plot.
        """

        cube = self.image[:,:,:]
        if threshold is not None:
            cube = np.where(cube > threshold, cube, 0)
        profile = np.nansum(cube, axis=(1,2)) / self._beam_area_pix()

        return profile

    def plot_line(self,x_axis="velocity", threshold=None, ax=None, **kwargs):
        """
        Plot the line profile of the cube.

        Args:
            x_axis (str): The axis to plot on.
            threshold (float): The threshold for the line profile.
            ax (matplotlib.axes.Axes): The axes to plot on.
            **kwargs: Additional keyword arguments for the plot.
        """

        if ax is None:
            ax = plt.gca()

        if x_axis == "channel":
            x = np.arange(self.nv)
        elif x_axis == "freq":
            x = self.nu
        else:
            x = self.velocity

        p = self.get_line_profile(threshold=threshold)

        ax.plot(x, p, **kwargs)

        dv = np.abs(self.velocity[2]-self.velocity[1])
        print("Integrated line flux =", (np.sum(p) - 0.5*(p[0]+p[-1]))*dv)

        return

    # -- computing various "moments"
    def get_moment_map(self, moment=0, v0=0, M0_threshold=None, M8_threshold=None, threshold=None, iv_support=None, v_minmax = None):
        """
        Calculate the moment map of the cube.

        We use the same convention as CASA : moment 8 is peak flux, moment 9 is peak velocity
        This returns the moment maps in physical units, ie:
        - M0 is the integrated line flux (Jy/beam . km/s)
        - M1 is the average velocity (km/s)
        - M2 is the velocity dispersion (km/s)
        - M8 is the peak intensity
        - M9 is the velocity of the peak

        Args:
            moment (int): The moment to calculate.
            v0 (float): The central velocity.
            M0_threshold (float): The threshold for the M0 moment map.
            M8_threshold (float): The threshold for the M8 moment map.
            threshold (float): The threshold for the moment map.
            iv_support (list): The indices of the velocity channels to support.
            v_minmax (list): The minimum and maximum velocities to plot.
        """

        if v0 is None:
            v0 = 0

        cube = np.copy(self.image)
        dv = (self.velocity[1] - self.velocity[0])
        v = self.velocity - v0

        if threshold is not None:
            cube = np.where(cube > threshold, cube, np.nan)

        if v_minmax is not None:
            vmin = np.min(v_minmax)
            vmax = np.max(v_minmax)
            iv_support = np.array(np.where(np.logical_and((self.velocity > vmin),(self.velocity < vmax)))).ravel()
            print("Selecting channels:", iv_support)

        if iv_support is not None:
            v = v[iv_support]
            cube = cube[iv_support,:,:]

        M0 = np.nansum(cube, axis=0) * dv
        M8 = np.max(cube, axis=0)

        if moment in [1, 2]:
            M1 = np.nansum(cube[:, :, :] * v[:, np.newaxis, np.newaxis], axis=0) * dv / M0

        if moment == 0:
            M=M0

        if  moment == 1:
            M=M1 + v0

        if moment == 2:
            # avoid division by 0 or neg values in sqrt
            thr = np.nanpercentile(M0[np.where(M0>0)],0.01)
            M0[np.where(M0<thr)]=np.nan
            M = np.sqrt(np.nansum(np.power(cube[:, :, :] * (v[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :]),2), axis=0) * dv / M0 )

        if moment == 8:
            M = M8

        if moment == 9:
            M = v[0] + dv * np.argmax(cube, axis=0)

        if M0_threshold is not None:
            M = np.ma.masked_where(M0 < M0_threshold, M)

        if M8_threshold is not None:
            M = np.ma.masked_where(M8 < M8_threshold, M)

        return M


    def get_high_pass_filter_map(self, moment=None, w0=None, gamma=0,  **kwargs):
        """
        Perform a Gaussian high pass filter on a moment map.
        w0 is in arcsec, with an optional radial stretch exponent gamma

        Args:
            moment (int): The moment to use for the high pass filter.
            w0 (float): The width of the Gaussian filter in arcsec.
            gamma (float): The radial stretch exponent.
            **kwargs: Additional keyword arguments for the moment map.
        """

        from scipy.ndimage.filters import gaussian_filter

        if w0 is None:
            raise ValueError("Please provide w0 for filter in sarcsec")

        # Original data
        M = self.get_moment_map(moment=moment,  **kwargs)
        X, Y = self.xaxis.copy(), self.xaxis.copy()


        if gamma != 0:
            # Stretch the original grid by desired radial stretch
            R = np.hypot(X, Y)
            x = X * (R**(gamma))
            y = Y * (R**(gamma))

            # Prepare interp function of input data on original grid
            f = interpolate.interp2d(X, Y, M, kind='cubic')

            # Interpolate input data onto radially stretched grid
            stretched_map = f(x, y)
        else:
            # No strectching
            stretched_map = M

        # Perform convolution on warped data and subtract result to get residual
        background = gaussian_filter(stretched_map, sigma=w0/self.pixelscale)
        highpass_residual = stretched_map - background

        if gamma != 0:
            # We reinterpolate back
            f = interpolate.interp2d(x, y, highpass_residual, kind='cubic')
            output_map = f(X,Y)
        else:
            output_map = highpass_residual

        return output_map


    def get_fwhm(self, v0=0, M0_threshold=None):
        """
        Get the line FWHM of the cube.

        Args:
            v0 (float): The central velocity.
            M0_threshold (float): The threshold for the M0 moment map.
        """

        M2 = get_moment_map(self, moment=2, v0=v0, M0_threshold=M0_threshold)

        return np.sqrt(8*np.log(2)) * M2

    def get_vturb(self, v0=0, M0_threshold=None, threshold=None, mol_weight=None):
        """
        Get the turbulent linewidth of the cube.

        Args:
            v0 (float): The central velocity.
            M0_threshold (float): The threshold for the M0 moment map.
        """

        if mol_weight is None:
            raise ValueError("mol_weight needs to be provided")

        M2 = self.get_moment_map(moment=2, v0=v0, M0_threshold=M0_threshold, threshold=threshold)
        Tb = self._Jybeam_to_Tb(self.get_moment_map(moment=8, v0=v0, M0_threshold=M0_threshold, threshold=threshold))

        mH = 1.007825032231/sc.N_A
        cs2 = sc.k * Tb / (mol_weight * mH)

        return np.sqrt(8*np.log(2)* M2**2 - 2*cs2)


    def get_std(self,taper=0):
        """
        Get the standard deviation of the cube.

        Args:
            taper (float): The taper for the beam.
        """

        im1 =self.image[0,0:30,0:30]
        im2 =self.image[-1,0:30,0:30]

        # This is ugly copy and paste
        # --- Convolution by taper
        if taper < 1e-6: # 0 taper
            taper = None

        if taper is not None:
            if taper < self.bmaj:
                print("taper is smaller than bmaj=", self.bmaj)
                delta_bmaj = self.pixelscale * FWHM_to_sigma
            else:
                delta_bmaj = np.sqrt(taper ** 2 - self.bmaj ** 2)
                bmaj = taper
            if taper < self.bmin:
                print("taper is smaller than bmin=", self.bmin)
                delta_bmin = self.pixelscale * FWHM_to_sigma
            else:
                delta_bmin = np.sqrt(taper ** 2 - self.bmin ** 2)
                bmin = taper

            sigma_x = delta_bmin / self.pixelscale * FWHM_to_sigma  # in pixels
            sigma_y = delta_bmaj / self.pixelscale * FWHM_to_sigma  # in pixels

            print("Original beam = ", self.bmaj, self.bmin)
            print("tapper =", delta_bmaj, delta_bmin, self.bpa)
            #print(
            #    "beam = ",
            #    np.sqrt(self.bmaj ** 2 + delta_bmaj ** 2),
            #    np.sqrt(self.bmin ** 2 + delta_bmin ** 2),
            #)
            self.bmaj = taper
            self.bmin = taper
            print("Updated beam = ", self.bmaj, self.bmin)

            beam = Gaussian2DKernel(sigma_x, sigma_y, self.bpa * np.pi / 180)

            im1 = convolve_fft(im1, beam)
            im2 = convolve_fft(im2, beam)

            # Correcting for beam area
            im1 *= taper**2/(self.bmin * self.bmaj)
            im2 *= taper**2/(self.bmin * self.bmaj)

        # compute std deviation in image, assumes that no primary beam correction has been applied
        self.std = np.nanstd([im1,im2])

        print("Std=",self.std)

        return

    # -- Functions to deal the synthesized beam.
    def _beam_area(self):
        """Beam area in arcsec^2"""
        return np.pi * self.bmaj * self.bmin / (4.0 * np.log(2.0))

    def _beam_area_str(self):
        """Beam area in steradian^2"""
        return self._beam_area() * arcsec ** 2

    def _pixel_area(self):
        """Pixel area in arcsec^2"""
        return self.pixelscale ** 2

    def _beam_area_pix(self):
        """Beam area in pix^2."""
        return self._beam_area() / self._pixel_area()

    @property
    def beam(self):
        """Returns the beam parameters in ("), ("), (deg)."""
        return self.bmaj, self.bmin, self.bpa

    def _Jybeam_to_Tb(self, im, RJ=False):
        """Convert flux converted from Jy/beam to K using full Planck law."""
        im2 = np.nan_to_num(im)
        nu = self.restfreq

        if RJ:
            Tb = abs(im2) * sc.c**2 / (2*nu**2*sc.k * 1e26 * self._beam_area_str())
        else:
            exp_m1 = 1e26 * self._beam_area_str() * 2.0 * sc.h * nu ** 3 / (sc.c ** 2 * abs(im2))
            hnu_kT = np.log1p(exp_m1 + 1e-10)
            Tb = sc.h * nu / (sc.k * hnu_kT)

        return np.ma.where(im2 >= 0.0, Tb, -Tb)


    def make_cut(self, x0,y0,x1,y1,z=None,num=None):
        """
        Make a cut in image 'z' along a line between (x0,y0) and (x1,y1)
        x0, y0,x1,y1 are pixel coordinates
        """

        if z is None:
            z = self.image

        if num is not None:
            # Extract the values along the line, using cubic interpolation
            x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
            zi = ndimage.map_coordinates(z, np.vstack((y,x)))

        else:
            # Extract the valuees along the line at the pixel spacing
            length = int(np.hypot(x1-x0, y1-y0))
            x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
            zi = z[y.astype(np.int), x.astype(np.int)]

        return x,y,zi


    def explore(self,num=None):
        """
        Quickly explore the cube.
           - plot line profile
           - plot peak brightness
           - comvert frequency to line and molecule name

        Args:
            num (int): The figure number.
        """

        # Quick preview function of a cube

        filename = self.filename
        k = filename.find('spw')
        spw = filename[k:k+5]

        freq = self.restfreq/1e9

        # Storing line, molecule and object names
        self.line = line_list[freq]
        self.mol = self.line[0:self.line.find(" ")]
        self.object = self.header['OBJECT']

        # Making preview plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), num=num,  clear=False)
        plt.subplots_adjust(hspace=0.01,wspace=0.01)


        # line profile
        ax = axes[0]
        self.plot_line(ax=ax)

        ax.text(0.02,0.95,self.object,
                horizontalalignment='left',size=12,
                color="black",
                transform=ax.transAxes,
                )


        ax.text(0.02,0.9,self.line,
                horizontalalignment='left',size=12,
                color="black",
                transform=ax.transAxes,
                )

        label=f"$\\nu=${freq:<10.6f}Ghz"
        ax.text(0.98,0.95,label,
                horizontalalignment='right',size=12,
                color="black",
                transform=ax.transAxes,
                )

        ax.text(0.98,0.9,spw,
                horizontalalignment='right',size=12,
                color="black",
                transform=ax.transAxes,
                )

        # Moment 8
        ax=axes[1]
        self.plot(ax=ax,moment=8, Tb=True) #32, 23, 20

        ax.text(0.02,0.95,"Peak brightness",
                horizontalalignment='left',size=12,
                color="white",
                transform=ax.transAxes,
                )

        return


# spatalogue astropy api is currently broken
#Splatalogue.query_lines(f*(1-1e-6)*u.GHz, f*(1+1e-6)*u.GHz)
# doing it by hand for now
line_list = {
    265.886431 : "HCN v=0 J=3-2"                    ,
    265.759438 : "c-HCCCH v=0 4(4,1)-3(3,0)"        ,
    265.289650 : "CH3OH v t=0 6(1,5)-5(2,3)"        ,
    264.2701294 : "H2CO 10(1,9)-10(1,10)"            ,
    262.2569057 : "SO2 v=0 11(3,9)-11(2,10)"         ,
    262.064986 : "CCH v=0 N=3-2, J=5/2-3/2, F=3-2" ,
    262.004260 : "CCH v=0 N=3-2, J=7/2-5/2, F=4-3" ,
    261.843721 : "SO 3Σ v=0 7(6)-6(5)"              ,
    251.900495 : "CH3OH v t=0 4(3,2)-4(2,3) +-"     ,
    251.825770 : "SO 3Σ v=0 5(6)-4(5)"              ,
    251.2105851 : "SO2 v=0 8(3,5)-8(2,6)"            ,
    250.816954 : "NO J=5/2-3/2, Ω=1/2-, F3/2-1/2"  ,
    330.5879653 : "13CO J=3-2",
    345.7959899 : "12CO J=3-2"
    #249.000000 : "Continuum"
}



def add_colorbar(mappable, shift=None, width=0.05, ax=None, trim_left=0, trim_right=0, side="right",**kwargs):
    """
    Add a color bar to a plot.

    Args:
        mappable (matplotlib.cm.ScalarMappable): The mappable object to add the color bar to.
        shift (float): The shift for the color bar.
        width (float): The width of the color bar.
        ax (matplotlib.axes.Axes): The axes to add the color bar to.
        trim_left (float): The left trim for the color bar.
        trim_right (float): The right trim for the color bar.
        side (str): The side of the color bar.
    """

    # creates a color bar that does not shrink the main plot or panel
    # only works for horizontal bars so far

    if ax is None:
        try:
            ax = mappable.axes
        except:
            ax = plt.gca()

    # Get current figure dimensions
    try:
        fig = ax.figure
        p = np.zeros([1,4])
        p[0,:] = ax.get_position().get_points().flatten()
    except:
        fig = ax[0].figure
        p = np.zeros([ax.size,4])
        for k, a in enumerate(ax):
            p[k,:] = a.get_position().get_points().flatten()
    xmin = np.amin(p[:,0]) ; xmax = np.amax(p[:,2]) ; dx = xmax - xmin
    ymin = np.amin(p[:,1]) ; ymax = np.amax(p[:,3]) ; dy = ymax - ymin

    if side=="top":
        if shift is None:
            shift = 0.2
        cax = fig.add_axes([xmin + trim_left, ymax + shift * dy, dx - trim_left - trim_right, width * dy])
        cax.xaxis.set_ticks_position('top')
        return fig.colorbar(mappable, cax=cax, orientation="horizontal",**kwargs)
    elif side=="right":
        if shift is None:
            shift = 0.05
        cax = fig.add_axes([xmax + shift*dx, ymin, width * dx, dy])
        cax.xaxis.set_ticks_position('top')
        return fig.colorbar(mappable, cax=cax, orientation="vertical",**kwargs)
