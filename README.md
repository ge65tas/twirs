# twirs
Rust port of the [astrometric plate solving package twirl](https://github.com/lgrcia/twirl). \
Currently, no source detection or catalog query is included. This only calculates the transformation between a set of detected sources and a provided catalog. To that end, asterisms of three or four stars are built, hashed, and then matched. \
Since it is faithful to the original, most documentation on the chosen algorithm applies.

## Documentation
Use `cargo doc --open` to read the package documentation.

## Acknowledgements
This is a port of [twirl](https://github.com/lgrcia/twirl) (L. J. Garcia), part of the image processing pipeline [prose](https://github.com/lgrcia/prose): \
Garcia, L. J. et al. (2022). prose: a Python framework for modular astronomical images processing. MNRAS, vol. 509, no. 4, pp. 4817–4828, 2022. [doi:10.1093/mnras/stab3113](https://academic.oup.com/mnras/article-abstract/509/4/4817/6414007).

As the original, it follows the algorithm: \
Lang, D. et al. (2010). _Astrometry.net: Blind Astrometric Calibration of Arbitrary Astronomical Images_. The Astronomical Journal, 139(5), pp.1782–1800. [doi:10.1088/0004-6256/139/5/1782](https://iopscience.iop.org/article/10.1088/0004-6256/139/5/1782).

Parts of the code, particularly in `ndarray_utils.rs`, were taken from or inspired by [nshare](https://github.com/rust-cv/nshare) (Rust Computer Vision).

For all licenses of dependencies, look into the subfolder `licenses` and particularly `license.html`.  
This file was automatically created using [cargo-about](https://github.com/EmbarkStudios/cargo-about) (Embark Studios).