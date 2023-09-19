"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
setup(
    name="pearlsim",
    version="0.1.0",
    description="Pebble-Explicit Advanced Reactor Learning Simulator, a ML-based approach to explicit PBR modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iankolaja/PEARLSim",
    author="I. Kolaja",
    author_email="ikolaja@berkeley.edu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="nuclear, data, exfor, endf, evaluated",  # Optional
    packages=find_packages(exclude=["tests"]),  # Required
    python_requires=">=3.7, <4",
    install_requires=["numpy", "pandas"],  # Optional
    extras_require={  # Optional
        "dev": ["check-manifest"],
        "test": ["coverage"],
    }
)
