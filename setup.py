"""Setup file for anomasota.

Later
1. https://docs.python.org/3/distutils/setupscript.html
2. https://packaging.python.org/en/latest/flow/

"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import List

from setuptools import find_packages, setup


ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"
REQUIREMENTS_DIR = ROOT_DIR / "requirements"


def _load_module(name: str):
    """Load a Python module from one of the modules in the directory `src`. 

    Args:
        name (str): Path to the module to load relative to the directory of this file.

    Returns:
        _type_: _description_
    """
    spec = spec_from_file_location(name=name, location=str(SRC_DIR / name))
    module = module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(module)  # type: ignore
    return module


def _get_version() -> str:
    """Get version from `anomasota.__init__.__version__`."""
    module = _load_module(name="anomasota/__init__.py")
    return module.__version__


VERSION = _get_version()
README = (ROOT_DIR / "README.md").read_text(encoding="utf8")
INSTALL_REQUIRES = (REQUIREMENTS_DIR / "base.txt").read_text(encoding="utf8").splitlines()
EXTRAS_REQUIRE = {}


setup(
    name="anomasota",
    version=_get_version(),
    author="jpcbertoldo",
    author_email="jpcbertoldo@minesparis.psl.eu",
    description="ANOMAly detection in State-Of-The-Art (anomasota)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/jpcbertoldo/anomasota",
    license="MIT License. Copyright (c) 2022 Joao P C Bertoldo. See LICENSE file for more details.",
    python_requires=">=3.7",
    package_dir={'': "src"},
    packages=find_packages("src"),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    package_data={},
    entry_points={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
