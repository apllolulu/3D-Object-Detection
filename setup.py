from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='box overlaps',
    ext_modules=cythonize('./utils/box_overlaps.pyx')
)

# solution for potential error related to numpy/arrayobject.h
#export CFLAGS="-I /home/ora/anaconda3/lib/python3.6/site-packages/numpy/core/include $CFLAGS"
