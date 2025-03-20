# casa_cube

casa_cube is a python package that provides an interface to data cubes generates by CASA or Gildas.

[![Documentation Status](https://readthedocs.org/projects/casa-cube/badge/?version=latest)](https://casa-cube.readthedocs.io/en/latest/)

It allows the user to perform simple tasks such plotting given channel maps, moment maps, line profile in various units, correcting for cloud extinction, reconvolving with a beam taper, triming a cube ...
The syntax is similar to pymcfost to perform quick and easy comparison with models.

## Installation

### Using pip
```
pip install casa_cube
```

### From the git repo
```
git clone https://github.com/cpinte/casa_cube.git
cd pymcfost
pip install .
```

To install in developer mode: (i.e. so that code changes here are immediately available without needing to repeat the above step):

```
pip install -e .
```
