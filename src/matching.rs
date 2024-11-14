//! Matching functions and main interface.

use std::fmt::Debug;

use itertools::Itertools;
use log::{debug, error, info};
use nalgebra::{Matrix3, MatrixXx2, RowVector2};
use ndarray::{s, Array2, ArrayView2, Axis};

use crate::geometry::transform_points;
use crate::ndarray_utils::{argmin, min_axis, norm_axis, IntoNdarray2};
use crate::quads::QuadAsterism;
use crate::triangles::TriangleAsterism;
use crate::wcs::Wcs;
use crate::Float;

/// Count the number of cross matches between two sets of 2D points.
///
/// # Arguments:
/// - `(coords1, coord2)`: Two sets of points. The number of points can differ.
/// - `tolerance`: Tolerance of the match. A good default is `1e-3`.
///
/// # Example:
/// ```
/// # use ndarray::array;
/// # use twirs::matching::count_cross_match;
/// let coords1 = array![[1.,2.], [3.,4.], [5.,6.]];
/// let coords2 = array![[1.,2.], [3.,4.], [6.,7.], [8., 9.]];
/// assert_eq!(count_cross_match(coords1.view(), coords2.view(), 0.), 2)
/// ```
pub fn count_cross_match<F: Float>(
    coords1: ArrayView2<F>,
    coords2: ArrayView2<F>,
    tolerance: F,
) -> usize {
    let coords1 = coords1.insert_axis(Axis(1));
    let coords2 = coords2.insert_axis(Axis(0));
    let diff = &coords1 - &coords2;
    let norm = norm_axis(diff.view(), Axis(2));
    let min = min_axis(norm.view(), Axis(0));
    min.fold(
        0,
        |acc, elem| if *elem <= tolerance { acc + 1 } else { acc },
    )
}

/// Finds the closest matches between two sets of 2D points.
///
/// # Arguments:
/// - `(coords1, coord2)`: Two sets of points. The number of points can differ.
/// - `tolerance`: Tolerance of the match, given in `coords1` points units. A good default is `10`.
///
/// # Example:
/// ```
/// # use ndarray::array;
/// # use twirs::matching::cross_match;
/// let coords1 = array![[3.,4.], [1.,2.], [5.,6.]];
/// let coords2 = array![[1.,2.], [3.,4.], [6.,7.]];
/// assert_eq!(cross_match(coords1.view(), coords2.view(), 10.), vec![[0,1],[1,0],[2,2]])
/// ```
pub fn cross_match<F: Float>(
    coords1: ArrayView2<F>,
    coords2: ArrayView2<F>,
    tolerance: F,
) -> Vec<[usize; 2]> {
    let len2 = coords2.shape()[0];
    let mut matches = Vec::new();

    for (i, point1) in coords1.axis_iter(Axis(0)).enumerate() {
        let diff: Vec<F> = coords2
            .axis_iter(Axis(0))
            .flat_map(|point2| &point1 - &point2)
            .collect_vec();
        let diff = Array2::from_shape_vec((len2, 2), diff).unwrap();
        let distances = norm_axis(diff.view(), Axis(1));
        let closest = argmin(distances.view());
        if distances[closest] < tolerance {
            matches.push([i, closest])
        }
    }

    matches
}

/// Generalizes over possible asterisms.
pub trait Asterism<F: Float>: Clone + Default {
    /// Type of the hashes.
    type Hashes: Clone + Debug;
    /// Type of the underlying asterism, e.g. a triangle.
    type Polygons: Clone + Debug + Into<Self::Matrix> + Send + Sync;
    /// Type of the point matrices used in order to calculate the transformations.
    type Matrix: Clone + Debug + Default;

    /// Calculate the hashes of all possible asterisms.
    fn hashes(&self, points: ArrayView2<F>) -> (Self::Hashes, Vec<Self::Polygons>);

    /// Find the matches between the asterisms.
    fn find_matches(
        hashes_pixels: Self::Hashes,
        hashes_radecs: Self::Hashes,
        tolerance: F,
    ) -> Vec<[usize; 2]>;

    /// Get the transformation matrix between the two sets of points.
    fn get_transformation_matrix(
        xy1: Self::Matrix,
        xy2: Self::Matrix,
    ) -> Result<Matrix3<F>, &'static str>;
}

/// The central struct of this library.
///
/// Use this in order to build options for plate solving.
/// For more details, check the module-level documentation.
#[derive(Clone, Debug)]
pub struct Twirs<F: Float, A: Asterism<F>> {
    /// List of pixels. Shape `(n_points, 2)`.
    pixels: Array2<F>,
    /// List of sky coordinates. Shape `(n_radecs, 2)`.
    radecs: Array2<F>,
    /// Choice of asterism.
    asterism: A,
    /// Hash matching tolerance.
    hash_tolerance: F,
    /// Match counting tolerance.
    tolerance: F,
    /// Minimum number of matches required, in fraction of amount of `pixels` given.
    min_match: Option<F>,
}

impl<F: Float + num_traits::Float> Twirs<F, TriangleAsterism<F>> {
    /// Use triangle asterisms.
    /// Use `with_` functions to set parameters.
    ///
    /// # Arguments
    /// -`pixels`: List of pixels. Shape `(n_points, 2)`.\
    /// -`radecs`: List of sky coordinates. Shape `(n_radecs, 2)`.
    pub fn triangles<A>(pixels: A, radecs: A) -> Self
    where
        A: IntoNdarray2<Out = Array2<F>>,
    {
        Self {
            pixels: pixels.into_ndarray2(),
            radecs: radecs.into_ndarray2(),
            asterism: TriangleAsterism::default(),
            tolerance: F::from_f64(5.).unwrap(),
            hash_tolerance: F::from_f64(0.1).unwrap(),
            min_match: None,
        }
    }

    /// Set the minimum angle of the triangles.
    pub fn with_min_angle(mut self, min_angle: F) -> Self {
        self.asterism.min_angle = min_angle;
        self
    }
}

impl<F: Float + num_traits::Float> Twirs<F, QuadAsterism> {
    /// Use quad asterisms.
    /// Use `with_` functions to set parameters.
    ///
    /// # Arguments
    /// -`pixels`: List of pixels. Shape `(n_points, 2)`.\
    /// -`radecs`: List of sky coordinates. Shape `(n_radecs, 2)`.
    pub fn quads<A>(pixels: A, radecs: A) -> Self
    where
        A: IntoNdarray2<Out = Array2<F>>,
    {
        Self {
            pixels: pixels.into_ndarray2(),
            radecs: radecs.into_ndarray2(),
            asterism: QuadAsterism,
            tolerance: F::from_f64(5.).unwrap(),
            hash_tolerance: F::from_f64(0.1).unwrap(),
            min_match: None,
        }
    }
}

impl<F, A> Twirs<F, A>
where
    F: Float,
    A: Asterism<F>,
{
    /// Create a new instance using default options.
    /// This allows for creating an instance using a generic [`Asterism`].
    /// If you want to use triangles, use [`triangles`](Twirs::triangles()), and for quads, use [`quads`](Twirs::quads()).
    pub fn new<N>(pixels: N, radecs: N) -> Self
    where
        N: IntoNdarray2<Out = Array2<F>>,
    {
        Self {
            pixels: pixels.into_ndarray2(),
            radecs: radecs.into_ndarray2(),
            asterism: A::default(),
            tolerance: F::from_f64(5.).unwrap(),
            hash_tolerance: F::from_f64(0.1).unwrap(),
            min_match: None,
        }
    }

    /// Set the hash matching tolerance.
    pub fn with_hash_tolerance(mut self, tolerance: F) -> Self {
        self.hash_tolerance = tolerance;
        self
    }

    /// Set the match counting tolerance.
    pub fn with_tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the minimum number of sources to match.
    pub fn with_min_match(mut self, min_match: F) -> Self {
        self.min_match = Some(min_match);
        self
    }

    /// Find the transformation matrix for an image given pixel and some unordered sky coordinates.
    ///
    /// # Returns
    /// If no match could be found inside the tolerance, returns `None`.\
    /// Otherwise, returns the 3x3 matrix \(T)\ that satisfies
    /// \[ T \cdot R^T \approx P^T, \]
    /// where \(R\) is the longest 1-padded list of catalog coordinates and \(P\) of sources that satisfy this relation.
    pub fn find_transform(self) -> Option<Matrix3<F>> {
        info!("Computing hashes.");
        let (hashes_pixels, mut asterism_pixels) = self.asterism.hashes(self.pixels.view());
        let (hashes_radecs, mut asterism_radecs) = self.asterism.hashes(self.radecs.view());

        let mut matches = Vec::new();
        let n_pixels = self.pixels.shape()[0];

        info!("Computing hash matches.");
        let pairs = A::find_matches(hashes_pixels, hashes_radecs, self.hash_tolerance);
        info!("Computing transformations for {} pairs.", pairs.len());
        for [i, j] in &pairs {
            let trafo = match A::get_transformation_matrix(
                asterism_radecs[*j].clone().into(),
                asterism_pixels[*i].clone().into(),
            ) {
                Ok(trafo) => trafo,
                Err(_) => continue,
            };

            let test = transform_points(self.radecs.clone(), trafo.into_ndarray2());
            let match_count = count_cross_match(self.pixels.view(), test.view(), self.tolerance);
            matches.push(match_count);

            if let Some(min_match) = self.min_match {
                if F::from_usize(match_count).unwrap()
                    >= min_match * F::from_usize(n_pixels).unwrap()
                {
                    break;
                }
            }
        }

        if matches.is_empty() {
            None
        } else {
            info!("Calculated transformation.");

            let argmax = matches
                .iter()
                .enumerate()
                .max_by(|(_, value0), (_, value1)| value0.cmp(value1))
                .map(|(idx, _)| idx)
                .unwrap();
            debug!(
                "Best transformation matches {} of {} sources.",
                matches[argmax], n_pixels
            );
            if let Some(min_match) = self.min_match {
                if F::from_usize(matches[argmax]).unwrap()
                    < min_match * F::from_usize(n_pixels).unwrap()
                {
                    error!("Matched less than the minimum number of sources!")
                }
            }

            let [i, j] = pairs[argmax];
            A::get_transformation_matrix(
                asterism_radecs.remove(j).into(),
                asterism_pixels.remove(i).into(),
            )
            .ok()
        }
    }

    /// Compute the WCS solution for an image given pixel and some unordered sky coordinates.
    ///
    /// This calls [`find_transform`](Twirs::find_transform()) and additionally calculates a
    /// [`Wcs`] transformation.
    pub fn compute_wcs(self) -> Option<Wcs<F>> {
        let radecs = self.radecs.clone();
        let pixels = self.pixels.clone();

        let trafo = self.find_transform()?;
        let radecs_xy = transform_points(radecs.clone(), trafo.into_ndarray2());
        let matches = cross_match(pixels.view(), radecs_xy.view(), F::from_f64(10.).unwrap());
        let (radecs_traf, pixels_traf): (Vec<RowVector2<F>>, Vec<RowVector2<F>>) = matches
            .into_iter()
            .map(|[i, j]| (radecs.slice(s![j, ..]), pixels.slice(s![i, ..])))
            .map(|(radec, pixel)| {
                (
                    RowVector2::new(radec[0], radec[1]),
                    RowVector2::new(pixel[0], pixel[1]),
                )
            })
            .unzip();

        if radecs_traf.is_empty() {
            return None
        }

        let radecs_traf = MatrixXx2::from_rows(&radecs_traf);
        let pixels_traf = MatrixXx2::from_rows(&pixels_traf);
        Wcs::from_points(pixels_traf, radecs_traf)
    }
}

#[cfg(feature = "parallel")]
pub use parallel::*;

#[cfg(feature = "parallel")]
mod parallel {
    use super::*;
    use crate::geometry::pad;
    use nalgebra::{Matrix2, Vector2, SVD};
    use rayon::prelude::*;

    impl<F, A> Twirs<F, A>
    where
        F: Float,
        A: Asterism<F>,
    {
        /// Find the transformation matrix for an image given pixel and some unordered sky coordiantes, in parallel.
        ///
        /// Also see [`find_transform`](Twirs::find_transform()) for more details.
        ///
        /// **Warning: This function ignores the `min_matches` option.**
        pub fn find_transform_par(self) -> Option<Matrix3<F>> {
            info!("Computing hashes.");
            let (hashes_pixels, mut asterism_pixels) = self.asterism.hashes(self.pixels.view());
            let (hashes_radecs, mut asterism_radecs) = self.asterism.hashes(self.radecs.view());

            info!("Computing hash matches.");
            let pairs = A::find_matches(hashes_pixels, hashes_radecs, self.hash_tolerance);
            info!("Computing transformations for {} pairs.", pairs.len());
            let matches = pairs.par_iter().map(|[i, j]| {
                let trafo = match A::get_transformation_matrix(
                    asterism_radecs[*j].clone().into(),
                    asterism_pixels[*i].clone().into(),
                ) {
                    Ok(trafo) => trafo,
                    Err(_) => return 0,
                };

                let test = transform_points(self.radecs.clone(), trafo.into_ndarray2());
                count_cross_match(self.pixels.view(), test.view(), self.tolerance)
            });

            info!("Calculated transformation.");

            let max = matches.enumerate().max_by_key(|(_, x)| *x)?;
            debug!(
                "Best transformation matches {} of {} sources.",
                max.1,
                self.pixels.shape()[0]
            );
            if let Some(min_match) = self.min_match {
                if F::from_usize(max.1).unwrap()
                    < min_match * F::from_usize(self.pixels.shape()[0]).unwrap()
                {
                    error!("Matched less than the minimum number of sources!")
                }
            }

            let [i, j] = pairs[max.0];
            A::get_transformation_matrix(
                asterism_radecs.remove(j).into(),
                asterism_pixels.remove(i).into(),
            )
            .ok()
        }

        /// Compute the WCS solution for an image given pixel and some unordered sky coordinates, in parallel.
        ///
        /// Also see [`compute_wcs`](Twirs::compute_wcs()) for more details.
        ///
        /// **Warning: This function ignores the `min_matches` option.**
        pub fn compute_wcs_par(self) -> Option<Wcs<F>> {
            let radecs = self.radecs.clone();
            let pixels = self.pixels.clone();

            let trafo = self.find_transform_par()?;
            let radecs_xy = transform_points(radecs.clone(), trafo.into_ndarray2());
            let matches = cross_match(pixels.view(), radecs_xy.view(), F::from_f64(10.).unwrap());
            let (radecs_traf, pixels_traf): (Vec<RowVector2<F>>, Vec<RowVector2<F>>) = matches
                .into_iter()
                .map(|[i, j]| (radecs.slice(s![j, ..]), pixels.slice(s![i, ..])))
                .map(|(radec, pixel)| {
                    (
                        RowVector2::new(radec[0], radec[1]),
                        RowVector2::new(pixel[0], pixel[1]),
                    )
                })
                .unzip();

            if radecs_traf.is_empty() {
                return None
            }

            let radecs_traf = MatrixXx2::from_rows(&radecs_traf);
            let mut pixels_traf = MatrixXx2::from_rows(&pixels_traf);
            let (pixels_x, pixels_y): (Vec<F>, Vec<F>) =
                pixels_traf.row_iter().map(|r| (r[0], r[1])).unzip();
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

            pixels_traf
                .row_iter_mut()
                .for_each(|mut r| r -= crpix.transpose());
            let radecs_pad = pad(radecs_traf);
            let pixels_pad = pad(pixels_traf);
            let svd = SVD::new(pixels_pad, true, true);
            let trafo: Matrix3<F> = svd.solve(&radecs_pad, F::from_f64(0.).unwrap()).ok()?;

            let wcs = Wcs {
                crpix: crpix + Vector2::new(F::from_f64(1.).unwrap(), F::from_f64(1.).unwrap()),
                crval: Vector2::new(trafo.m31, trafo.m32),
                cd: Matrix2::new(trafo.m11, trafo.m21, trafo.m12, trafo.m22),
            };
            Some(wcs)
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_abs_diff_eq;
    use itertools::Itertools;
    use nalgebra::Vector2;
    use ndarray::{array, Array1, Array2};
    use ndarray_rand::{rand_distr::Normal, RandomExt};
    use numpy::{PyArray2, ToPyArray, PyArrayMethods};
    use pyo3::{prelude::*, types::IntoPyDict};
    use rand::{distributions::Uniform, Rng};

    #[test]
    fn count_cross_match() {
        let points1 = Array2::random((50, 2), Uniform::new(0., 1.));
        let points2 = Array2::random((30, 2), Uniform::new(0., 1.));
        let tol = 0.1;

        let count = super::count_cross_match(points1.view(), points2.view(), tol);

        let count_py: usize = Python::with_gil(|py| {
            let twirl_match = py.import_bound("twirl.match").unwrap();

            let points1_py = points1.to_pyarray_bound(py);
            let points2_py = points2.to_pyarray_bound(py);

            let kwargs = [("tol", tol)].into_py_dict_bound(py);
            let count = twirl_match
                .call_method("count_cross_match", (points1_py, points2_py), Some(&kwargs))
                .unwrap()
                .extract()
                .unwrap();

            // py.run_bound("del twirl.match", None, None).unwrap();
            count
        });

        assert_eq!(count, count_py);
    }

    #[test]
    fn cross_match() {
        let points1 = Array2::random((50, 2), Uniform::new(0., 1.));
        let points2 = Array2::random((30, 2), Uniform::new(0., 1.));
        let tol = 0.1;

        let matched = super::cross_match(points1.view(), points2.view(), tol);
        let matched = Array2::from_shape_vec(
            (matched.len(), 2),
            matched
                .into_iter()
                .flatten()
                .map(|x| x as isize)
                .collect_vec(),
        )
        .unwrap();

        let matched_py = Python::with_gil(|py| {
            let twirl_match = py.import_bound("twirl.match").unwrap();

            let points1_py = points1.to_pyarray_bound(py);
            let points2_py = points2.to_pyarray_bound(py);

            let kwargs = [("tolerance", tol)].into_py_dict_bound(py);
            let matched_py = twirl_match
                .call_method("cross_match", (points1_py, points2_py), Some(&kwargs))
                .unwrap()
                .downcast::<PyArray2<isize>>()
                .unwrap().clone();

            // py.run_bound("del twirl.match", None, None).unwrap();

            matched_py.to_owned_array()
        });

        dbg!(&matched);
        assert_eq!(matched, matched_py);
    }

    #[test]
    fn find_transform_triangle() {
        let mut rng = rand::thread_rng();

        let shape = (25, 2);
        let angle: f64 = rng.gen();
        let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
        let offset = Array1::random(2, Uniform::new(0., 1.));

        dbg!(angle, &offset);

        let pixels = Array2::random(shape, Uniform::new(0., 1.));
        let radecs_vec = pixels
            .rows()
            .into_iter()
            .flat_map(|r| rot.dot(&r) + offset.view())
            .collect_vec();
        let radecs = Array2::from_shape_vec(shape, radecs_vec).unwrap()
            + Array2::<f64>::random(shape, Normal::new(0., 0.001).unwrap());

        let twirl = Twirs::triangles(pixels.clone(), radecs.clone())
            .with_min_match(0.7)
            .with_hash_tolerance(0.02)
            .with_tolerance(12.);
        let trafo = twirl.find_transform().unwrap();

        let trafo_dyn_py = Python::with_gil(|py| {
            let twirl_match = py.import_bound("twirl.match").unwrap();

            let pixels_py = pixels.to_pyarray_bound(py);
            let radecs_py = radecs.to_pyarray_bound(py);

            let kwargs = [("asterism", 3)].into_py_dict_bound(py);
            let trafo_py = twirl_match
                .call_method("find_transform", (radecs_py, pixels_py), Some(&kwargs))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap().clone();

            // py.run_bound("del twirl.match", None, None).unwrap();

            trafo_py.readonly().as_matrix().clone_owned()
        });
        let mut trafo_py = Matrix3::zeros();
        for (d, o) in trafo_dyn_py.into_iter().zip(trafo_py.iter_mut()) {
            *o = *d;
        }

        let test = transform_points(radecs.clone(), trafo.into_ndarray2());
        let match_count = super::count_cross_match(pixels.view(), test.view(), 5.);
        let test_py = transform_points(radecs, trafo_py.into_ndarray2());
        let match_count_py = super::count_cross_match(pixels.view(), test_py.view(), 5.);

        assert_eq!(match_count, match_count_py);
    }

    #[test]
    fn find_transform_quad() {
        let mut rng = rand::thread_rng();

        let shape = (25, 2);
        let angle: f64 = rng.gen();
        let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
        let offset = Array1::random(2, Uniform::new(0., 10.));

        dbg!(angle, &offset);

        let pixels = Array2::random(shape, Uniform::new(0., 100.));
        let radecs_vec = pixels
            .rows()
            .into_iter()
            .flat_map(|r| rot.dot(&r) + offset.view())
            .collect_vec();
        let radecs = Array2::from_shape_vec(shape, radecs_vec).unwrap()
            + Array2::<f64>::random(shape, Normal::new(0., 1.).unwrap());

        let twirl = Twirs::quads(pixels.clone(), radecs.clone())
            .with_min_match(0.7)
            .with_hash_tolerance(0.02)
            .with_tolerance(1.);
        let trafo = twirl.find_transform().unwrap();

        let trafo_dyn_py = Python::with_gil(|py| {
            let twirl_match = py.import_bound("twirl.match").unwrap();

            let pixels_py = pixels.to_pyarray_bound(py);
            let radecs_py = radecs.to_pyarray_bound(py);

            let trafo_py = twirl_match
                .call_method("find_transform", (radecs_py, pixels_py), None)
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap().clone();

            // py.run_bound("del twirl.match", None, None).unwrap();

            trafo_py.readonly().as_matrix().clone_owned()
        });
        let mut trafo_py = Matrix3::zeros();
        for (d, o) in trafo_dyn_py.into_iter().zip(trafo_py.iter_mut()) {
            *o = *d;
        }

        let test = transform_points(radecs.clone(), trafo.into_ndarray2());
        let match_count = super::count_cross_match(pixels.view(), test.view(), 5.);
        let test_py = transform_points(radecs, trafo_py.into_ndarray2());
        let match_count_py = super::count_cross_match(pixels.view(), test_py.view(), 5.);
        dbg!(match_count, match_count_py);

        assert_eq!(match_count, match_count_py);
    }

    #[test]
    fn compute_wcs() {
        let mut rng = rand::thread_rng();

        let shape = (15, 2);
        let angle: f64 = rng.gen();
        let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
        let offset = Array1::random(2, Uniform::new(0., 1.));

        dbg!(angle, &offset);

        let pixels = Array2::random(shape, Uniform::new(0., 1.));
        let radecs_vec = pixels
            .rows()
            .into_iter()
            .flat_map(|r| rot.dot(&r) + offset.view())
            .collect_vec();
        let radecs = Array2::from_shape_vec(shape, radecs_vec).unwrap()
            + Array2::<f64>::random(shape, Normal::new(0., 0.001).unwrap());
        dbg!(&pixels);

        let twirl = Twirs::quads(pixels.clone(), radecs.clone());
        let wcs = twirl.compute_wcs_par().unwrap();

        for (pixel, radec) in pixels.rows().into_iter().zip(radecs.rows()) {
            let pixel = Vector2::new(pixel[0], pixel[1]);
            let radec = Vector2::new(radec[0], radec[1]);
            assert_abs_diff_eq!(radec, wcs.pixel_to_world(pixel), epsilon = 1e-2);
        }
    }
}
