//! Simple implementation of the World Coordinate System standard.

use nalgebra::{Matrix2, Matrix3, MatrixXx2, Vector2, SVD};

use crate::geometry::pad;
use crate::Float;

/// Relevant data for WCS transformations from FITS headers.
#[derive(Clone, Debug, PartialEq)]
pub struct Wcs<F: Float> {
    /// Reference pixel.
    pub crpix: Vector2<F>,
    /// Coordinate value at `crpix`.
    pub crval: Vector2<F>,
    /// Linear transformation matrix.
    pub cd: Matrix2<F>,
}

impl<F: Float> Wcs<F> {
    /// Create a new instance.
    pub fn new(crpix: Vector2<F>, crval: Vector2<F>, cd: Matrix2<F>) -> Self {
        Self { crpix, crval, cd }
    }

    /// Find the WCS solution by matching lists of pixel and sky coordinates.
    pub fn from_points(mut pixels: MatrixXx2<F>, radecs: MatrixXx2<F>) -> Option<Self> {
        let (pixels_x, pixels_y): (Vec<F>, Vec<F>) =
            pixels.row_iter().map(|r| (r[0], r[1])).unzip();
        let xmin = pixels_x
            .iter()
            .copied()
            .reduce(|f1, f2| f1.min(f2))
            .unwrap();
        let xmax = pixels_x.into_iter().reduce(|f1, f2| f1.max(f2)).unwrap();
        let ymin = pixels_y
            .iter()
            .copied()
            .reduce(|f1, f2| f1.min(f2))
            .unwrap();
        let ymax = pixels_y.into_iter().reduce(|f1, f2| f1.max(f2)).unwrap();
        let crpix = Vector2::new(
            (xmin + xmax) / F::from_f64(2.).unwrap(),
            (ymin + ymax) / F::from_f64(2.).unwrap(),
        );

        pixels
            .row_iter_mut()
            .for_each(|mut r| r -= crpix.transpose());
        let radecs_pad = pad(radecs);
        let pixels_pad = pad(pixels);
        let svd = SVD::new(pixels_pad, true, true);
        let trafo: Matrix3<F> = svd.solve(&radecs_pad, F::from_f64(0.).unwrap()).ok()?;

        let wcs = Wcs {
            crpix: crpix + Vector2::new(F::from_f64(1.).unwrap(), F::from_f64(1.).unwrap()),
            crval: Vector2::new(trafo.m31, trafo.m32),
            cd: Matrix2::new(trafo.m11, trafo.m21, trafo.m12, trafo.m22),
        };
        Some(wcs)
    }

    /// Transforms from pixel to sky coordinate space.
    pub fn pixel_to_world(&self, pixel: Vector2<F>) -> Vector2<F> {
        let pixel = pixel + Vector2::new(F::from_f64(1.).unwrap(), F::from_f64(1.).unwrap());
        self.crval + self.cd * (pixel - self.crpix)
    }

    /// Transforms from sky coordinate to pixel space.
    pub fn world_to_pixel(&self, world_coordinate: Vector2<F>) -> Vector2<F> {
        let cd_inv = self.cd.try_inverse().unwrap();

        let pixel = self.crpix + cd_inv * (world_coordinate - self.crval);
        pixel - Vector2::new(F::from_f64(1.).unwrap(), F::from_f64(1.).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use numpy::ToPyArray;
    use pyo3::{
        types::{IntoPyDict, PyAnyMethods, PyList, PyModule, PyTuple},
        Py, PyAny, Python,
    };
    use rand::Rng;

    use super::*;

    #[test]
    fn astropy() {
        let mut rng = rand::thread_rng();

        let crval = Vector2::new(rng.gen_range(0.0..45.0), rng.gen_range(0.0..45.0));
        let crpix = Vector2::new(rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0));
        let cd = Matrix2::new(
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
        );

        let pixel_to_transf = Vector2::new(rng.gen_range(0.0..10.0), rng.gen_range(0.0..10.0));
        let radec_to_transf = Vector2::new(rng.gen_range(0.0..45.0), rng.gen_range(0.0..45.0));

        let wcs = Wcs::new(crpix, crval, cd);
        let pixel = wcs.world_to_pixel(radec_to_transf);
        let radec = wcs.pixel_to_world(pixel_to_transf);

        let (x, y, ra, dec) = Python::with_gil(|py| {
            let fun: Py<PyAny> = PyModule::from_code_bound(
                py,
                "import astropy.wcs

def init_wcs(crval, crpix, cd):
    w = astropy.wcs.WCS(naxis=2)
    w.wcs.crval = crval
    w.wcs.crpix = crpix
    w.wcs.cd = cd
    return w",
                "test",
                "test",
            )
            .unwrap()
            .getattr("init_wcs")
            .unwrap()
            .into();

            let crval_py = crval
                .to_pyarray_bound(py)
                .call_method0("transpose")
                .unwrap()
                .call_method1("__getitem__", (0,))
                .unwrap();
            let crpix_py = crpix
                .to_pyarray_bound(py)
                .call_method0("transpose")
                .unwrap()
                .call_method1("__getitem__", (0,))
                .unwrap();
            let cd_py = cd.to_pyarray_bound(py);
            let wcs = fun.call1(py, (crval_py, crpix_py, cd_py)).unwrap();

            let sky = wcs
                .call_method1(py, "pixel_to_world", (pixel_to_transf.x, pixel_to_transf.y))
                .unwrap();
            let sky = sky.downcast_bound::<PyList>(py).unwrap();
            let ra = sky.get_item(0).unwrap()
                .call_method0("__float__")
                .unwrap()
                .extract::<f64>()
                .unwrap();
            let dec = sky.get_item(1).unwrap()
                .call_method0("__float__")
                .unwrap()
                .extract::<f64>()
                .unwrap();

            let pix = wcs
                .call_method1(
                    py,
                    "world_to_pixel_values",
                    (radec_to_transf.x, radec_to_transf.y),
                )
                .unwrap();
            let pix = pix.downcast_bound::<PyTuple>(py).unwrap();
            let x = pix.get_item(0).unwrap()
                .call_method0("__float__")
                .unwrap()
                .extract::<f64>()
                .unwrap();
            let y = pix.get_item(1).unwrap()
                .call_method0("__float__")
                .unwrap()
                .extract::<f64>()
                .unwrap();

            (x, y, ra, dec)
        });

        // Compare to `astropy`
        assert_abs_diff_eq!(radec.x, ra, epsilon = 1e-6);
        assert_abs_diff_eq!(radec.y, dec, epsilon = 1e-6);
        assert_abs_diff_eq!(pixel.x, x, epsilon = 1e-6);
        assert_abs_diff_eq!(pixel.y, y, epsilon = 1e-6);
    }

    #[test]
    fn consistency_check() {
        let wcs = Wcs {
            crval: Vector2::new(2.711529441199E+01, -3.925398447545E+01),
            crpix: Vector2::new(5.065191000000E+02, 4.892484000000E+02),
            cd: Matrix2::new(
                1.672682044534E-04,
                1.996643749806E-06,
                -9.963899403011E-08,
                1.729743106508E-04,
            ),
        };

        let wc = wcs.pixel_to_world(Vector2::new(0., 0.));
        let px = wcs.world_to_pixel(wc);

        assert_abs_diff_eq!(px.x, 0., epsilon = 1e-10);
        assert_abs_diff_eq!(px.y, 0., epsilon = 1e-10);
    }
}
