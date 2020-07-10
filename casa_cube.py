import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
import scipy.ndimage


FWHM_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2)))
arcsec = np.pi / 648000
default_cmap = "inferno"


class Cube:
    def __init__(self, filename, **kwargs):

        self.filename = os.path.normpath(os.path.expanduser(filename))
        self._read(**kwargs)

    def _read(self):
        try:
            hdu = fits.open(self.filename)
            self.header = hdu[0].header
            self.image = np.ma.masked_array(hdu[0].data)

            if self.image.ndim == 4:
                self.image = self.image[0, :, :, :]

            # Read a few keywords in header
            try:
                self.object = hdu[0].header['OBJECT']
            except:
                self.object = ""
            self.unit = hdu[0].header['BUNIT']

            # pixel info
            self.nx = hdu[0].header['NAXIS1']
            self.ny = hdu[0].header['NAXIS2']
            self.pixelscale = hdu[0].header['CDELT2'] * 3600 # arcsec
            self.cx = hdu[0].header['CRPIX1']
            self.cy = hdu[0].header['CRPIX2']
            self.x_ref = hdu[0].header['CRVAL1']  # coordinate
            self.y_ref = hdu[0].header['CRVAL2']
            self.FOV = np.maximum(self.nx, self.ny) * self.pixelscale

            halfsize = np.asarray([self.nx, self.ny]) / 2 * self.pixelscale
            self.extent = [halfsize[0], -halfsize[0], -halfsize[1], halfsize[1]]

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
            except:
                self.restfreq = hdu[0].header['RESTFREQ']  # gildas format
            self.wl = sc.c / self.restfreq
            try:
                self.velocity_type = hdu[0].header['CTYPE3']
                self.CRPIX3 = hdu[0].header['CRPIX3']
                self.CRVAL3 = hdu[0].header['CRVAL3']
                self.CDELT3 = hdu[0].header['CDELT3']
                if self.velocity_type == "VELO-LSR": # gildas
                    self.velocity = self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)
                    self.nu = self.restfreq * (1 - self.velocity * 1000 / sc.c)
                elif self.velocity_type == "VRAD":  # casa format : v en km/s
                    self.velocity = (self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)) / 1000
                    self.nu = self.restfreq * (1 - self.velocity * 1000 / sc.c)
                elif self.velocity_type == "FREQ": # Hz
                    self.nu = self.CRVAL3 + self.CDELT3 * (np.arange(1, self.nv + 1) - self.CRPIX3)
                    self.velocity = (-(self.nu - self.restfreq) / self.restfreq * sc.c / 1000.0)  # km/s
                else:
                    raise ValueError("Velocity type is not recognised:", self.velocity_type)
            except:
                pass

            # beam
            try:
                self.bmaj = hdu[0].header['BMAJ'] * 3600.0  # arcsec
                self.bmin = hdu[0].header['BMIN'] * 3600.0
                self.bpa = hdu[0].header['BPA']
            except:
                # make an average of all the records ...
                self.bmaj = hdu[1].data[0][0]
                self.bmin = hdu[1].data[0][1]
                self.bpa = hdu[1].data[0][2]

            hdu.close()
        except OSError:
            print('cannot open', self.filename)
            return ValueError

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
        resample=0,
        bmaj=None,
        bmin=None,
        bpa=None,
        taper=None,
        colorbar_label=True,
        M0_threshold=None,
        vlabel_position="bottom",
        shift_dx=0,
        shift_dy=0,
    ):

        if ax is None:
            ax = plt.gca()

        unit = self.unit
        xlabel = r'$\Delta$ RA ["]'
        ylabel = r'$\Delta$ Dec ["]'

        if self.nv == 1:  # continuum image
            is_cont = True
            if self.image.ndim > 2:
                im = self.image[0, :, :]
            else:
                im = self.image
            _color_scale = 'log'
        elif moment is not None:
            is_cont = False
            im = self.get_moment_map(moment=moment, v0=v0, M0_threshold=M0_threshold)
            _color_scale = 'lin'
        else:
            is_cont = False
            # -- Selecting channel corresponding to a given velocity
            if dv is not None:
                v = v0 + dv

            if v is not None:
                print(v)
                iv = np.abs(self.velocity - v).argmin()
                print("Selecting channel #", iv)
            im = self.image[iv, :, :]
            _color_scale = 'lin'

        if Tb:
            im = self._Jybeam_to_Tb(im)
            unit = "K"
            _color_scale = 'lin'

        # --- Convolution by taper
        if taper is not None:
            if taper < self.bmaj:
                print("taper is smaller than bmaj=", self.bmaj)
                delta_bmaj = self.pixelscale * FWHM_to_sigma
            else:
                delta_bmaj = np.sqrt(
                    taper ** 2 - self.bmaj ** 2
                )  # sigma will be 1 pixel
                bmaj = taper
            if taper < self.bmin:
                print("taper is smaller than bmin=", self.bmin)
                delta_bmin = self.pixelscale * FWHM_to_sigma
            else:
                delta_bmin = np.sqrt(taper ** 2 - self.bmin ** 2)
                bmin = taper

            sigma_x = delta_bmin / self.pixelscale * FWHM_to_sigma  # in pixels
            sigma_y = delta_bmaj / self.pixelscale * FWHM_to_sigma  # in pixels

            print("beam = ", self.bmaj, self.bmin)
            print("tapper =", delta_bmaj, delta_bmin, self.bpa)
            print(
                "beam = ",
                np.sqrt(self.bmaj ** 2 + delta_bmaj ** 2),
                np.sqrt(self.bmin ** 2 + delta_bmin ** 2),
            )

            beam = Gaussian2DKernel(sigma_x, sigma_y, self.bpa * np.pi / 180)
            im = convolve_fft(im, beam)

        # --- resampling
        if resample > 0:
            mask = scipy.ndimage.zoom(im.mask * 1, resample, order=3)
            im = scipy.ndimage.zoom(im.data, resample, order=3)
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

        # -- set up the color scale
        if color_scale == 'log':
            norm = colors.LogNorm(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'lin':
            norm = colors.Normalize(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'sqrt':
            norm = colors.PowerNorm(0.5, vmin=fmin, vmax=fmax, clip=True)
        else:
            raise ValueError("Unknown color scale: " + color_scale)

        if cmap is None:
            if moment in [1, 9]:
                cmap = "RdBu_r"
            else:
                cmap = default_cmap

        # --- Making the actual plot
        image = ax.imshow(
            im,
            norm=norm,
            extent=self.extent - np.asarray([shift_dx, shift_dx, shift_dy, shift_dy]),
            origin='lower',
            cmap=cmap,
            alpha=alpha,
            interpolation="bicubic",
        )

        if limit is not None:
            limits = [limit, -limit, -limit, limit]

        if limits is not None:
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[2], limits[3])

        if not no_xlabel:
            ax.set_xlabel(xlabel)
        if not no_ylabel:
            ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        # -- Color bar
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(image, cax=cax)

            # cax,kw = mpl.colorbar.make_axes(ax)
            # cb = plt.colorbar(image,cax=cax, **kw)
            formatted_unit = unit.replace("-1", "$^{-1}$").replace("-2", "$^{-2}$")

            if colorbar_label:
                if moment == 0:
                    cb.set_label("Flux [" + formatted_unit + ".km.s$^{-1}$]")
                elif moment in [1, 9]:
                    cb.set_label("Velocity [km.s$^{-1}]$")
                elif moment == 2:
                    cb.set_label("Velocity dispersion [km$^2$.s$^{-2}$]")
                else:
                    if Tb:
                        cb.set_label("T$_\mathrm{b}$ [" + formatted_unit + "]")
                    else:
                        cb.set_label("Flux [" + formatted_unit + "]")
            plt.sca(ax)  # we reset the main axis

        # -- Adding velocity
        if vlabel_position == "top":
            y_vlabel = 0.85
        else:
            y_vlabel = 0.1
        if not no_vlabel:
            if (moment is None) and not is_cont:
                if v0 is None:
                    ax.text(
                        0.5,
                        y_vlabel,
                        f"v={self.velocity[iv]:<4.2f}$\,$km/s",
                        horizontalalignment='center',
                        color="white",
                        transform=ax.transAxes,
                    )
                else:
                    ax.text(
                        0.5,
                        y_vlabel,
                        f"$\Delta$v={self.velocity[iv] -v0:<4.2f}$\,$km/s",
                        horizontalalignment='center',
                        color="white",
                        transform=ax.transAxes,
                    )

        # --- Adding beam
        if plot_beam:
            dx = 0.125
            dy = 0.125

            # In case the beam is wrong in the header, when can pass the correct one
            if bmaj is None:
                bmaj = self.bmaj
            if bmin is None:
                bmin = self.bmin
            if bpa is None:
                bpa = self.bpa

            beam = Ellipse(
                ax.transLimits.inverted().transform((dx, dy)),
                width=bmin,
                height=bmaj,
                angle=-bpa,
                fill=True,
                color="grey",
            )
            ax.add_patch(beam)

        return image

    # -- computing various "moments"
    def get_moment_map(self, moment=0, v0=0, M0_threshold=None):
        """
        We use the same comvention as CASA : moment 8 is peak flux, moment 9 is peak velocity
        This returns the moment maps in physical units, ie:
         - M0 is the integrated line flux (Jy/beam . km/s)
         - M1 is the average velocity [km/s]
         - M2 is the velocity dispersion [km/s]
        """

        if v0 is None:
            v0 = 0

        cube = np.copy(self.image)
        dv = self.velocity[1] - self.velocity[0]
        v = self.velocity - v0

        if moment <= 2:
            M0 = np.sum(cube, axis=0) * np.abs(dv)

        if M0_threshold is not None:
            M0 = np.ma.masked_less(M0, M0_threshold)

        if moment in [1, 2]:
            M1 = (
                np.sum(cube[:, :, :] * v[:, np.newaxis, np.newaxis], axis=0)
                * np.abs(dv)
                / M0
            )

        if moment == 0:
            return M0
        elif moment == 1:
            return M1

        if moment == 2:
            M2 = np.sqrt(
                np.sum(
                    cube[:, :, :]
                    * (v[:, np.newaxis, np.newaxis] - M1[np.newaxis, :, :]) ** 2,
                    axis=0,
                )
                * np.abs(dv)
                / M0
            )
            return M2

        if moment == 8:
            M8 = np.max(cube, axis=0)
            return M8

        if moment == 9:
            M9 = self.velocity[0] + dv * np.argmax(cube, axis=0)
            return M9

    # -- Functions to deal the synthesized beam.
    def _beam_area(self):
        """Beam area in arcsec^2"""
        return np.pi * self.bmaj * self.bmin / (4.0 * np.log(2.0))

    def _beam_area_str(self):
        """Beam area in steradian^2"""
        return self._beam_area() * arcsec ** 2

    def _pixel_area(self):
        return self.pixelscale ** 2

    def _beam_area_pix(self):
        """Beam area in pix^2."""
        return self._beam_area() / self._pixel_area()

    @property
    def beam(self):
        """Returns the beam parameters in ["], ["], [deg]."""
        return self.bmaj, self.bmin, self.bpa

    def _Jybeam_to_Tb(self, im):
        """Convert flux converted from Jy/beam to K using full Planck law."""
        im2 = np.nan_to_num(im)
        nu = self.restfreq

        exp_m1 = (
            1e26 * self._beam_area_str() * 2.0 * sc.h * nu ** 3 / (sc.c ** 2 * abs(im2))
        )
        hnu_kT = np.log1p(exp_m1 + 1e-10)
        Tb = sc.h * nu / (sc.k * hnu_kT)

        return np.ma.where(im2 >= 0.0, Tb, -Tb)
