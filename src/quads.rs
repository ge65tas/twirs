//! Asterism of four stars.

use std::{
    array,
    ops::{Deref, DerefMut, Index, IndexMut, Sub},
};

use itertools::Itertools;
use kiddo::float::{kdtree::KdTree, distance::SquaredEuclidean};
use nalgebra::{matrix, Matrix3, Matrix4x2, Vector2, SVD};
use ndarray::{s, Array2, ArrayView2, Axis, ShapeError};

use crate::geometry::{pad, projection_sum, u1u2};
use crate::matching::Asterism;
use crate::ndarray_utils::{argmax, max_axis, norm_axis};
use crate::Float;

#[derive(Debug, Clone, PartialEq)]
pub struct Quad<F: Float> {
    points: [Vector2<F>; 4],
}

impl<F: Float> Quad<F> {
    pub(crate) fn new(
        point1: Vector2<F>,
        point2: Vector2<F>,
        point3: Vector2<F>,
        point4: Vector2<F>,
    ) -> Self {
        Self {
            points: [point1, point2, point3, point4],
        }
    }
}

impl<F: Float> Quad<F> {
    /// Sort the points by their largest seperation to the other points.
    pub(crate) fn sort_by_distance(self) -> Self {
        let arr: Array2<F> = self.try_into().unwrap();

        // get all differences between points
        let diff = &arr.view().insert_axis(Axis(2)) - &arr.t().insert_axis(Axis(0));
        // get all distances between points
        let distances = norm_axis(diff.view(), Axis(1));
        // all highest distances
        let max = max_axis(distances.view(), Axis(0));
        // get index of point with highest distance to any other point (`A`)
        let max_arg = argmax(max.view());
        // get row with highest distance
        let largest_distances = distances.slice(s![.., max_arg]);
        // sort points by largest distance and move the last point to the second
        let sorted: Vec<F> = arr
            .axis_iter(Axis(0))
            .zip(largest_distances)
            .sorted_by(|(_, dist1), (_, dist2)| dist1.partial_cmp(dist2).expect("found nan"))
            .flat_map(|(v, _)| v)
            .copied() // TODO: Optimize by removing clones
            .collect_vec(); // TODO: Wrong order

        Quad::new(
            matrix![sorted[0]; sorted[1]],
            matrix![sorted[6]; sorted[7]],
            matrix![sorted[4]; sorted[5]],
            matrix![sorted[2]; sorted[3]],
        )
    }
}

impl<F: Float> Deref for Quad<F> {
    type Target = [Vector2<F>; 4];

    fn deref(&self) -> &Self::Target {
        &self.points
    }
}

impl<F: Float> DerefMut for Quad<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.points
    }
}

impl<F: Float> Index<usize> for Quad<F> {
    type Output = Vector2<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

impl<F: Float> IndexMut<usize> for Quad<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.points[index]
    }
}

impl<F: Float> IntoIterator for Quad<F> {
    type Item = Vector2<F>;

    type IntoIter = array::IntoIter<Vector2<F>, 4>;

    fn into_iter(self) -> Self::IntoIter {
        self.points.into_iter()
    }
}

impl<F: Float> TryFrom<Quad<F>> for Array2<F> {
    type Error = ShapeError;

    fn try_from(value: Quad<F>) -> Result<Self, Self::Error> {
        let vec = value
            .into_iter()
            .flat_map(|v| -> [F; 2] { v.into() })
            .collect_vec();
        Array2::from_shape_vec((4, 2), vec)
    }
}

impl<F: Float> From<Quad<F>> for Matrix4x2<F> {
    fn from(value: Quad<F>) -> Self {
        matrix![
            value[0][0], value[0][1];
            value[1][0], value[1][1];
            value[2][0], value[2][1];
            value[3][0], value[3][1];
        ]
    }
}

impl<F: Float> Sub<Vector2<F>> for Quad<F> {
    type Output = Quad<F>;

    fn sub(self, rhs: Vector2<F>) -> Self::Output {
        Quad::new(self[0] - rhs, self[1] - rhs, self[2] - rhs, self[3] - rhs)
    }
}

impl<F: Float> Sub<Vector2<F>> for &Quad<F> {
    type Output = Quad<F>;

    fn sub(self, rhs: Vector2<F>) -> Self::Output {
        Quad::new(self[0] - rhs, self[1] - rhs, self[2] - rhs, self[3] - rhs)
    }
}

/// Remove quads that have points outside a circle.
///
/// # Arguments
/// - `quads`: The quads to be filtered.
/// - `circle_tolerance`: How much outside the circle points may lie. A good default is 0.01.
fn good_quads<F: Float>(
    quads: impl Iterator<Item = Quad<F>>,
    circle_tolerance: F,
) -> impl Iterator<Item = Quad<F>> {
    quads.filter(move |quad| {
        let point1 = quad[0];
        let point2 = quad[1];
        let r = (point2 - point1).norm() / F::from_f64(2.).unwrap();
        let center = point1 + (point2 - point1) / F::from_f64(2.).unwrap();
        let shifted = quad - center;
        shifted
            .into_iter()
            .map(|v| v.norm())
            .all(|n| n <= r * (F::from_f64(1.).unwrap() + circle_tolerance))
    })
}

/// Calculate the hashes using asterisms of four stars.
///
/// For more information, look at [Lang et al. 2010](<https://iopscience.iop.org/article/10.1088/0004-6256/139/5/1782>).
#[derive(Clone, Debug)]
pub struct QuadAsterism;

impl QuadAsterism {
    /// Computes the hashes of the quads.
    fn quad_hash<F: Float>(quads: &[Quad<F>]) -> Array2<F> {
        quads
            .iter()
            .map(|quad| {
                let Quad {
                    points: [a, b, c, d],
                } = &quad;
                let norm = (b - a).norm();
                let (u1, u2) = u1u2(a, b);

                [
                    projection_sum(c, a, &u1) / norm,
                    projection_sum(d, a, &u1) / norm,
                    projection_sum(c, a, &u2) / norm,
                    projection_sum(d, a, &u2) / norm,
                ]
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn hashes_unsorted<F: Float>(points: ArrayView2<F>) -> (Array2<F>, Vec<Quad<F>>) {
        assert!(
            points.shape()[0] >= 4,
            "at least 4 points required to build quads"
        );

        let combinations = (0..points.shape()[0]).combinations(4);
        let quads = combinations.map(move |idx| {
            Quad::new(
                matrix![points[[idx[0], 0]]; points[[idx[0], 1]]],
                matrix![points[[idx[1], 0]]; points[[idx[1], 1]]],
                matrix![points[[idx[2], 0]]; points[[idx[2], 1]]],
                matrix![points[[idx[3], 0]]; points[[idx[3], 1]]],
            )
        });
        let ordered_quads = quads.map(|q| q.sort_by_distance());
        let good_ordered_quads: Vec<Quad<F>> =
            good_quads(ordered_quads, F::from_f64(0.01).unwrap()).collect();

        (
            Self::quad_hash(&good_ordered_quads),
            good_ordered_quads.clone(),
        )
    }
}

impl Default for QuadAsterism {
    fn default() -> Self {
        Self
    }
}

impl<F: Float + num_traits::float::FloatCore> Asterism<F> for QuadAsterism {
    type Hashes = Array2<F>;
    type Polygons = Quad<F>;
    type Matrix = Matrix4x2<F>;

    fn find_matches(
        hashes_pixels: Array2<F>,
        hashes_radecs: Array2<F>,
        tolerance: F, 
    ) -> Vec<[usize; 2]> {
        let pixel_tree: KdTree<F, usize, 4, 32, u32> = hashes_pixels
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, r)| ([r[0], r[1], r[2], r[3]], i))
            .collect();

        let pairs = hashes_radecs
            .axis_iter(Axis(0))
            .enumerate()
            .flat_map(|(i, row)| {
                let mut pairs = Vec::new();
                // TODO: Sorted better?
                let matches = pixel_tree.within_unsorted::<SquaredEuclidean>(
                    &[row[0], row[1], row[2], row[3]],
                    num_traits::float::FloatCore::powi(tolerance, 2),
                );
                for m in matches {
                    pairs.push([m.item, i]);
                }
                pairs
            })
            .collect();

        pairs
    }

    /// Computes the hashes of the quads formed by `points`.
    ///
    /// # Arguments
    /// - `points`: An matrix of shape (n_points, 2) representing the x and y coordinates of each point.
    fn hashes(&self, points: ArrayView2<F>) -> (Array2<F>, Vec<Quad<F>>) {
        let (hashes, good_ordered_quads) = Self::hashes_unsorted(points);

        let (quads_out, hashes_vec): (_, Vec<_>) = good_ordered_quads
            .into_iter()
            .zip(hashes.rows())
            .sorted_by(|(q1, _), (q2, _)| {
                (q1[1] - q1[0])
                    .norm()
                    .partial_cmp(&(q2[1] - q2[0]).norm())
                    .expect("found nan")
                    .reverse()
            })
            .unzip();

        let hashes_flat: Vec<[F; 4]> = hashes_vec
            .into_iter()
            .map(|r| [r[0], r[1], r[2], r[3]])
            .collect();

        let hashes_out = hashes_flat.into();

        (hashes_out, quads_out)
    }

    fn get_transformation_matrix(
        xy1: Matrix4x2<F>,
        xy2: Matrix4x2<F>,
    ) -> Result<Matrix3<F>, &'static str> {
        let xy1 = pad(xy1);
        let xy2 = pad(xy2);

        let svd = SVD::new(xy1, true, true);
        Ok(svd.solve(&xy2, F::from_f64(0.).unwrap())?.transpose())
    }
}

#[cfg(test)]
mod tests {
    use crate::ndarray_utils::IntoNdarray3;

    use super::*;
    use approx::assert_abs_diff_eq;
    use nalgebra::matrix;
    use ndarray::array;
    use ndarray_rand::rand_distr::Normal;
    use ndarray_rand::RandomExt;
    use numpy::{PyArray2, PyArray3, ToPyArray, PyArrayMethods};
    use pyo3::{prelude::*, types::PyTuple};
    use rand::distributions::Uniform;
    use rand::prelude::*;

    #[test]
    fn try_from() {
        let quad = Quad::new(
            matrix![1.; 2.],
            matrix![3.; 4.],
            matrix![5.; 6.],
            matrix![7.; 8.],
        );
        let arr: Array2<f64> = quad.try_into().unwrap();
        assert_eq!(arr, array![[1., 2.], [3., 4.], [5., 6.], [7., 8.]]);
    }

    #[test]
    fn order_points() {
        let mut rng = rand::thread_rng();

        let quads = (0..10)
            .map(|_| {
                Quad::<f64>::new(
                    Vector2::new(rng.gen(), rng.gen()),
                    Vector2::new(rng.gen(), rng.gen()),
                    Vector2::new(rng.gen(), rng.gen()),
                    Vector2::new(rng.gen(), rng.gen()),
                )
            })
            .collect_vec();

        let ordered_points = quads
            .clone()
            .into_iter()
            .map(|q| q.sort_by_distance())
            .collect_vec()
            .into_ndarray3();

        let ordered_points_py = Python::with_gil(|py| {
            let twirl_quads = py.import_bound("twirl.quads").unwrap();

            let arr = quads.into_ndarray3().to_pyarray_bound(py);
            let ordered_points_py = twirl_quads
                .call_method1("reorder", (arr,))
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap().clone();
            ordered_points_py.to_owned_array()
        });

        assert_eq!(ordered_points, ordered_points_py);
    }

    #[test]
    fn quad_hash() {
        let mut rng = rand::thread_rng();

        let quads = (0..4)
            .map(|_| {
                Quad::<f64>::new(
                    Vector2::new(rng.gen(), rng.gen()),
                    Vector2::new(rng.gen(), rng.gen()),
                    Vector2::new(rng.gen(), rng.gen()),
                    Vector2::new(rng.gen(), rng.gen()),
                )
            })
            .collect_vec();

        let hashes = QuadAsterism::quad_hash(&quads);

        let hashes_py = Python::with_gil(|py| {
            let twirl_quads = py.import_bound("twirl.quads").unwrap();

            let arr = quads.into_ndarray3().to_pyarray_bound(py);
            let hashes_quads_py = twirl_quads
                .call_method1("quad_hash", (arr,))
                .unwrap()
                .downcast::<PyTuple>()
                .unwrap().clone();
            let hashes_py = hashes_quads_py.get_item(0).unwrap().downcast::<PyArray2<f64>>().unwrap().clone();

            hashes_py.to_owned_array()
        });

        assert_abs_diff_eq!(hashes, hashes_py, epsilon = 1e-6);
    }

    #[test]
    fn hashes() {
        let points = Array2::random((10, 2), Uniform::new(0., 10.));

        let (hashes, quads) = QuadAsterism::hashes_unsorted(points.view());

        let (hashes_py, quads_py) = Python::with_gil(|py| {
            let fun: Py<PyAny> = PyModule::from_code_bound(
                py,
                "import numpy as np
import itertools
import twirl.quads as q

def hash(xy):
    quads_idxs = np.array(list(itertools.combinations(np.arange(xy.shape[0]), 4)))
    quads = xy[quads_idxs]
    ordered_quads = q.reorder(quads)
    good_ordered_quads = ordered_quads[q.good_quads(ordered_quads)]
    hashes = q.quad_hash(good_ordered_quads)[0]
    return hashes, good_ordered_quads",
                "test",
                "test",
            )
            .unwrap()
            .getattr("hash")
            .unwrap()
            .into();

            let arr = points.to_pyarray_bound(py);
            let hashes_quads_py = fun.call1(py, (arr,)).unwrap();
            let hashes_quads_py = hashes_quads_py.downcast_bound::<PyTuple>(py).unwrap();

            let hashes_py = hashes_quads_py.get_item(0).unwrap().downcast::<PyArray2<f64>>().unwrap().clone();
            let quads_py = hashes_quads_py.get_item(1).unwrap().downcast::<PyArray3<f64>>().unwrap().clone();

            let hashes_py = hashes_py.to_owned_array();
            let quads_py = quads_py.to_owned_array();

            (hashes_py, quads_py)
        });

        assert_abs_diff_eq!(hashes, hashes_py, epsilon = 1e-6);
        assert_abs_diff_eq!(quads.into_ndarray3(), quads_py, epsilon = 1e-6);
    }

    #[test]
    fn find_matches() {
        let points1 = Array2::random((20, 2), Uniform::new(0., 1.));
        let points2 =
            points1.clone() + Array2::<f64>::random((20, 2), Normal::new(0., 0.007).unwrap());

        let ast = QuadAsterism;
        let (hashes1, _) = ast.hashes(points1.view());
        let (hashes2, _) = ast.hashes(points2.view());

        let pairs = QuadAsterism::find_matches(hashes1.clone(), hashes2.clone(), 0.02);

        let pairs_py = Python::with_gil(|py| {
            let fun: Py<PyAny> = PyModule::from_code_bound(
                py,
                "from scipy.spatial import cKDTree

def pairs(hashes_pixels, hashes_radecs):
    tree_pixels = cKDTree(hashes_pixels)
    tree_radecs = cKDTree(hashes_radecs)
    pairs = []

    ball_query = tree_pixels.query_ball_tree(tree_radecs, r=0.02)

    for i, j in enumerate(ball_query):
        if len(j) > 0:
            pairs += [[i, k] for k in j]

    return pairs",
                "test",
                "test",
            )
            .unwrap()
            .getattr("pairs")
            .unwrap()
            .into();

            let hashes1_py = hashes1.to_pyarray_bound(py);
            let hashes2_py = hashes2.to_pyarray_bound(py);
            let pairs_py = fun.call1(py, (hashes1_py, hashes2_py)).unwrap();
            pairs_py.extract::<Vec<Vec<usize>>>(py).unwrap()
        });

        let mut pairs_sorted = pairs.into_iter().map(|a| a.to_vec()).collect_vec();
        pairs_sorted.sort_by_key(|x| x[0]);

        assert_eq!(pairs_sorted, pairs_py);
    }
}
