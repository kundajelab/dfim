import os, sys
import numpy
from setuptools import setup, Extension, find_packages

def main():
    setup(
        name = "dfim",
        version = "0.1",

        author = "Peyton Greenside",
        author_email = "pgreenside@gmail.com",

        install_requires = [ 'scipy', 'numpy>=1.11', 'pandas', 'deeplift',],

        extra_requires=['matplotlib', 'h5py', 'biopython'],

        packages= ['dfim'],

        description = ("Deep Feature Interaction Maps"),

        license = "GPL3",
        keywords = "feature interactions for deep learning models",
        url = "https://github.com/kundajelab/dfim",

        long_description="""
        DFIM discovers interactions between inputs to a deep learning model. Primary intended application to uncover interactions between sequence motifs in DNA or between mutations and surrounding sequence. By computing the difference in importance scores after perturbation of input we can disentangle which predictive features rely on each other versus act independently.
        """,
        classifiers=[
            "Programming Language :: Python :: 2",
            "Development Status :: 3 - Alpha",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        ],

    )

if __name__ == '__main__':
    main()

