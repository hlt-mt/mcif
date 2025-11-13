Installation
============

You can install the latest stable version from PyPI::

    pip install mcif

Or, to install from source::

    git clone https://github.com/hlt-mt/mcif.git
    cd mcif
    pip install .

Notice that, since some evaluation metrics run on GPU (e.g. COMET), you may want to first
setup your environment installing versions of torch with CUDA support for you GPU.

For development (with docs and testing tools)::

    pip install .[dev]
