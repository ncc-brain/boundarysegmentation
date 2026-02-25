"""
Shim package to expose third-party boundary segmentation modules as a standard
Python package for imports such as ``from boundry_segmentation.uboco_gebd``.

The directory already contains the implementation modules (e.g. ``uboco_gebd``),
but lacked an ``__init__`` so Python treated it as a plain folder. Creating this
file lets the existing import statements succeed without modifying upstream
code.
"""

# Nothing to initialize; the presence of this file is enough to make Python
# treat ``boundry_segmentation`` as a regular package.

