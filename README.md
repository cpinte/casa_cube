# casa_cube

casa_cube is a python package that provides an interface to data cubes generates by CASA or Gildas.
It allows the user to perform simple tasks such plotting given channel maps, moment maps, line profile in various units, correcting for cloud extinction, reconvolving with a beam taper, ...
The syntax is similar to pymcfost to perform quick and easy comparison with models.

## Installation:

```
git clone https://github.com/cpinte/casa_cube.git
cd casa_cube
python3 setup.py install
```

If you don't have the `sudo` rights, use `python3 setup.py install --user`.

To install in developer mode: (i.e. using symlinks to point directly
at this directory, so that code changes here are immediately available
without needing to repeat the above step):

```
 python3 setup.py develop
```

## Basic usage for 3D molecular line cube
```
import casa_cube as casa

obs = casa.Cube('HD_163296_CO_220GHz.robust_0.5_wcont.image.fits')
obs.plot(iv=0)
```
For a line cube one must specify either the channel number (iv=) or a moment map:
```
obs.plot(moment=1,cmap='RdBu_r')
```

## Basic usage for 2D fits image
```
import casa_cube as casa

obs = casa.Cube('HD169142_2015-05-03_Q_phi.fits',pixelscale=0.01225)
obs.plot()
```
