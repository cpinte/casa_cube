import os
import numpy as np
from astropy.io import fits
import scipy.constants as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.ndimage import zoom, map_coordinates
import cmasher as cmr


FWHM_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2)))
arcsec = np.pi / 648000
default_cmap = cmr.arctic


class Cube:
    def __init__(self, filename, only_header=False, correct_fct=None, unit=None, pixelscale=None, restfreq=None, zoom=None, **kwargs):

        self.filename = os.path.normpath(os.path.expanduser(filename))
        self._read(**kwargs, only_header=only_header, correct_fct=correct_fct, unit=unit, pixelscale=pixelscale, restfreq=restfreq, zoom=zoom)

    def _read(self, only_header=False, correct_fct=None, unit=None, pixelscale=None, instrument=None, restfreq=None, zoom=None):
        try:
            hdu = fits.open(self.filename)
            self.header = hdu[0].header

            # Read a few keywords in header
            try:
                self.object = hdu[0].header['OBJECT']
            except:
                self.object = ""

            try:
                self.unit = hdu[0].header['BUNIT']
            except:
                print("Warning : could not find unit")
                self.unit = ""

            if unit is not None:
                print("Warning : forcing unit")
                self.unit=unit

            if self.unit == "beam-1 Jy": # discminer format
                self.unit = "Jy/beam"

            if self.unit == "JY/PIXEL": # radmc format
                self.unit = "Jy pixel-1"


            # pixel info
            self.nx = hdu[0].header['NAXIS1']
            self.ny = hdu[0].header['NAXIS2']
            try:
                self.pixelscale = hdu[0].header['CDELT2'] * 3600 # arcsec
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
                    pixelscale = 7.36e-3
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
                elif instrument == "IRDIS_K2":
                    pixelscale = 12.267e-3
                elif instrument == "IRDIS_K3":
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
                        print("Warning : missing rest frequency")
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
                print("Warning : could not extract velocity")
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
                    print("Warning : missing beam")
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

                if correct_fct is not None:
                    self.image *= correct_fct[:,np.newaxis, np.newaxis]

                if zoom is not None:
                    print('Original size = ', self.image.shape)
                    self.image[np.isnan(self.image)] = 0.
                    self.image = zoom(self.image, zoom=[1, zoom, zoom])
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

        import copy

        image = copy.deepcopy(self.image)
        header = copy.deepcopy(self.header)

        ndim = image.ndim

        # ---- Spatial trimming
        if FOV is not None:
            cutout_pix = int(FOV / self.pixelscale)
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
        if (image.ndim == 4):
            image = image[p_min:p_max,iv_min:iv_max,iy_min:iy_max,ix_min:ix_max]
        elif (image.ndim == 3):
            image = image[iv_min:iv_max,iy_min:iy_max,ix_min:ix_max]
        else:
            raise ValueError("incorrect dimension in fits file")

        fits.writeto(os.path.normpath(os.path.expanduser(filename)),image.data, header, **kwargs)

        return


    def writeto(filename, image, header, **kwargs):
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
        interpolation=None,
        resample=0,
        bmaj=None,
        bmin=None,
        bpa=None,
        taper=None,
        colorbar_label=True,
        M0_threshold=None,
        M8_threshold=None,
        threshold = None,
        threshold_value = np.NaN,
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
        mJy=False
    ):
        """
        Plotting routine for continuum image, moment maps and channel maps.
        """


        if ax is None:
            ax = plt.gca()

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

            im = self.image[iv, :, :]
            _color_scale = 'lin'


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
            mask = zoom(im.mask * 1, resample, order=3)
            im = zoom(im.data, resample, order=3)
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
            norm = mcolors.LogNorm(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'lin':
            norm = mcolors.Normalize(vmin=fmin, vmax=fmax, clip=True)
        elif color_scale == 'sqrt':
            norm = mcolors.PowerNorm(0.5, vmin=fmin, vmax=fmax, clip=True)
        else:
            raise ValueError("Unknown color scale: " + color_scale)

        if cmap is None:
            if moment in [1, 9]:
                cmap = "RdBu_r"
            else:
                cmap = default_cmap

        if colors is not None:
            cmap = None


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
        extent = [-halfsize[0]*xaxis_factor-shift_dx, halfsize[0]*xaxis_factor-shift_dx, -halfsize[1]-shift_dy, halfsize[1]-shift_dy]
        if axes_unit.lower() == 'pixels' or axes_unit.lower() == 'pixel':
            extent = None

        self.extent = extent

        if threshold is not None:
            im = np.where(im > threshold, im, threshold_value)

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
            image = ax.contour(
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
            #divider = make_axes_locatable(ax)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            #cb = plt.colorbar(image, cax=cax, extend=colorbar_extend)
            cb = add_colorbar(image)

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
                        f"v={self.velocity[iv]:<4.2f}$\,$km/s",
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

    def get_line_profile(self,threshold=None, **kwargs):

        cube = self.image[:,:,:]
        if threshold is not None:
            cube = np.where(cube > threshold, cube, 0)
        profile = np.nansum(cube, axis=(1,2)) / self._beam_area_pix()

        return profile

    def plot_line(self,x_axis="velocity", threshold=None, **kwargs):

        if x_axis == "channel":
            x = np.arange(self.nv)
        elif x_axis == "freq":
            x = self.nu
        else:
            x = self.velocity

        p = self.get_line_profile(threshold=threshold)

        plt.plot(x, p, **kwargs)

        dv = np.abs(self.velocity[2]-self.velocity[1])
        print("Integrated line flux =", (np.sum(p) - 0.5*(p[0]+p[-1]))*dv)

        return

    # -- computing various "moments"
    def get_moment_map(self, moment=0, v0=0, M0_threshold=None, M8_threshold=None, threshold=None, iv_support=None, v_minmax = None):
        """
        We use the same convention as CASA : moment 8 is peak flux, moment 9 is peak velocity
        This returns the moment maps in physical units, ie:
         - M0 is the integrated line flux (Jy/beam . km/s)
         - M1 is the average velocity (km/s)
         - M2 is the velocity dispersion (km/s)
         - M8 is the peak intensity
         - M9 is the velocity of the peak
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

    def get_fwhm(self, v0=0, M0_threshold=None):

        M2 = get_moment_map(self, moment=2, v0=v0, M0_threshold=M0_threshold)

        return np.sqrt(8*np.log(2)) * M2

    def get_vturb(self, v0=0, M0_threshold=None, threshold=None, mol_weight=None):

        if mol_weight is None:
            raise ValueError("mol_weight needs to be provided")

        M2 = self.get_moment_map(moment=2, v0=v0, M0_threshold=M0_threshold, threshold=threshold)
        Tb = self._Jybeam_to_Tb(self.get_moment_map(moment=8, v0=v0, M0_threshold=M0_threshold, threshold=threshold))

        mH = 1.007825032231/sc.N_A
        cs2 = sc.k * Tb / (mol_weight * mH)

        return np.sqrt(8*np.log(2)* M2**2 - 2*cs2)


    def get_std(self):
        # compute std deviation in image, assumes that no primary beam correction has been applied
        self.std = np.nanstd(self.image[[0,-1],:,:])

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
        """Returns the beam parameters in ("), ("), (deg)."""
        return self.bmaj, self.bmin, self.bpa

    def _Jybeam_to_Tb(self, im):
        """Convert flux converted from Jy/beam to K using full Planck law."""
        im2 = np.nan_to_num(im)
        nu = self.restfreq

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
            zi = map_coordinates(z, np.vstack((y,x)))

        else:
            # Extract the values along the line at the pixel spacing
            length = int(np.hypot(x1-x0, y1-y0))
            x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
            zi = z[y.astype(np.int), x.astype(np.int)]

        return x,y,zi


def add_colorbar(mappable, shift=None, width=0.05, ax=None, trim_left=0, trim_right=0, side="right",**kwargs):
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
