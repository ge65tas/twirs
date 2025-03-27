//! Asterism of three stars.

use std::array;
use std::cmp::Ordering;
use std::ops::{Deref, DerefMut, Index, IndexMut, Sub};

use itertools::Itertools;
use kiddo::float::{distance::SquaredEuclidean, kdtree::KdTree};
use nalgebra::{matrix, vector, Matrix3, Matrix3x2, Vector2, SVD};
use ndarray::{Array2, ArrayView2};

use crate::geometry::pad;
use crate::matching::Asterism;
use crate::Float;

#[derive(Debug, Clone, PartialEq)]
pub struct Triangle<F: Float> {
    vertices: [Vector2<F>; 3],
}

impl<F: Float> Triangle<F> {
    pub(crate) fn new(vertex1: Vector2<F>, vertex2: Vector2<F>, vertex3: Vector2<F>) -> Self {
        Self {
            vertices: [vertex1, vertex2, vertex3],
        }
    }
}

impl<F: Float> Triangle<F> {
    /// Calculate the centroid.
    pub(crate) fn centroid(&self) -> Vector2<F> {
        let x = (self[0][0] + self[1][0] + self[2][0]) / F::from_f64(3.).unwrap();
        let y = (self[0][1] + self[1][1] + self[2][1]) / F::from_f64(3.).unwrap();
        vector![x, y]
    }

    /// Sort the vertices by their distance from the centroid.
    pub(crate) fn sort_by_distance(self) -> Self {
        let centroid = self.centroid();
        let distances = (&self - centroid)
            .into_iter()
            .map(|v| v.dot(&v).sqrt())
            .collect_vec();
        let mut indexed_vertices = self.vertices.into_iter().enumerate().collect_vec();
        indexed_vertices
            .sort_unstable_by(|(i, _), (j, _)| distances[*i].partial_cmp(&distances[*j]).unwrap());

        Triangle::new(
            indexed_vertices[0].1,
            indexed_vertices[1].1,
            indexed_vertices[2].1,
        )
    }

    /// Calculate all angles.
    pub(crate) fn _angles(&self) -> [F; 3] {
        let vec1 = self[1] - self[0];
        let vec2 = self[2] - self[1];
        let vec3 = self[0] - self[2];

        let norm1 = vec1.norm();
        let norm2 = vec2.norm();
        let norm3 = vec3.norm();

        let angle1 = ((norm2 * norm2 + norm3 * norm3 - norm1 * norm1)
            / (F::from_f64(2.0).unwrap() * norm2 * norm3))
            .acos();
        let angle2 = ((norm3 * norm3 + norm1 * norm1 - norm2 * norm2)
            / (F::from_f64(2.0).unwrap() * norm3 * norm1))
            .acos();
        let angle3 = ((norm1 * norm1 + norm2 * norm2 - norm3 * norm3)
            / (F::from_f64(2.0).unwrap() * norm1 * norm2))
            .acos();

        [angle1, angle2, angle3]
    }

    /// Calculate all angles, sort them by magnitude and discard the highest one.
    pub(crate) fn angles_sorted_discard(&self) -> [F; 2] {
        let vec1 = self[1] - self[0];
        let vec2 = self[2] - self[1];
        let vec3 = self[0] - self[2];

        let norm1 = vec1.norm();
        let norm2 = vec2.norm();
        let norm3 = vec3.norm();

        let angle1 = ((norm2 * norm2 + norm3 * norm3 - norm1 * norm1)
            / (F::from_f64(2.0).unwrap() * norm2 * norm3))
            .acos();
        let angle2 = ((norm3 * norm3 + norm1 * norm1 - norm2 * norm2)
            / (F::from_f64(2.0).unwrap() * norm3 * norm1))
            .acos();
        let angle3 = ((norm1 * norm1 + norm2 * norm2 - norm3 * norm3)
            / (F::from_f64(2.0).unwrap() * norm1 * norm2))
            .acos();

        let mut angles = [angle1, angle2, angle3];
        angles.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        [angles[0], angles[1]]
    }
}

impl<F: Float> Deref for Triangle<F> {
    type Target = [Vector2<F>; 3];

    fn deref(&self) -> &Self::Target {
        &self.vertices
    }
}

impl<F: Float> DerefMut for Triangle<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vertices
    }
}

impl<F: Float> Index<usize> for Triangle<F> {
    type Output = Vector2<F>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vertices[index]
    }
}

impl<F: Float> IndexMut<usize> for Triangle<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vertices[index]
    }
}

impl<F: Float> IntoIterator for Triangle<F> {
    type Item = Vector2<F>;

    type IntoIter = array::IntoIter<Vector2<F>, 3>;

    fn into_iter(self) -> Self::IntoIter {
        self.vertices.into_iter()
    }
}

impl<F: Float> Sub<Vector2<F>> for Triangle<F> {
    type Output = Triangle<F>;

    fn sub(self, rhs: Vector2<F>) -> Self::Output {
        Triangle::new(self[0] - rhs, self[1] - rhs, self[2] - rhs)
    }
}

impl<F: Float> Sub<Vector2<F>> for &Triangle<F> {
    type Output = Triangle<F>;

    fn sub(self, rhs: Vector2<F>) -> Self::Output {
        Triangle::new(self[0] - rhs, self[1] - rhs, self[2] - rhs)
    }
}

impl<F: Float> Sub<Vector2<F>> for &mut Triangle<F> {
    type Output = Triangle<F>;

    fn sub(self, rhs: Vector2<F>) -> Self::Output {
        Triangle::new(self[0] - rhs, self[1] - rhs, self[2] - rhs)
    }
}

impl<F: Float> From<Triangle<F>> for Matrix3x2<F> {
    fn from(value: Triangle<F>) -> Self {
        matrix![
            value[0][0], value[0][1];
            value[1][0], value[1][1];
            value[2][0], value[2][1];
        ]
    }
}

/// Orders the vertices of each triangle in a consistent manner.
///
/// Vertices are ordered by their distance to the centroid of the triangle.
fn order_points<F: Float>(
    triangles: impl Iterator<Item = Triangle<F>>,
) -> impl Iterator<Item = Triangle<F>> {
    triangles.map(|v| v.sort_by_distance())
}

/// Calculate the hashes using asterisms of three stars.
///
/// For more information, look at [Lang et al. 2010](<https://iopscience.iop.org/article/10.1088/0004-6256/139/5/1782>).
#[derive(Clone, Debug)]
pub struct TriangleAsterism<F: Float> {
    /// The minimum angle for all three angles of the triangles.
    pub min_angle: F,
}

impl<F: Float> Default for TriangleAsterism<F> {
    fn default() -> Self {
        Self {
            min_angle: F::from_f64(30f64.to_radians()).unwrap(),
        }
    }
}

impl<F: Float> TriangleAsterism<F> {
    /// Create a new instance from a minimum angle.
    pub fn new(min_angle: F) -> Self {
        Self { min_angle }
    }
}

impl<F: Default + Float + num_traits::float::FloatCore> Asterism<F> for TriangleAsterism<F> {
    type Hashes = Array2<F>;
    type Polygons = Triangle<F>;
    type Matrix = Matrix3x2<F>;

    fn find_matches(
        hashes_pixels: Array2<F>,
        hashes_radecs: Array2<F>,
        tolerance: F,
    ) -> Vec<[usize; 2]> {
        let mut pairs = Vec::new();

        let pixel_tree: KdTree<F, usize, 2, 32, u32> = hashes_pixels
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, r)| ([r[0], r[1]], i))
            .collect();

        for (i, row) in hashes_radecs.rows().into_iter().enumerate() {
            // TODO: Sorted better?
            let matches = pixel_tree.within_unsorted::<SquaredEuclidean>(
                &[row[0], row[1]],
                num_traits::float::FloatCore::powi(tolerance, 2),
            );
            for m in matches {
                pairs.push([m.item, i]);
            }
        }

        pairs
    }

    /// Computes the hashes of the triangles formed by `points`.
    ///
    /// # Arguments
    /// - `points`: An matrix of shape (n_points, 2) representing the x and y coordinates of each point.
    /// - `min_angle`: The minimum angle (in radians) that a triangle must have to be included in the hashes. A good default is 30 degrees.
    fn hashes(&self, points: ArrayView2<F>) -> (Array2<F>, Vec<Triangle<F>>) {
        assert!(
            points.shape()[0] >= 3,
            "at least 3 points required to build triangles"
        );

        let n_points = points.shape()[0];
        let combinations = (0..n_points).combinations(3);
        let triangles = combinations.map(|idx| {
            Triangle::new(
                matrix![points[[idx[0], 0]]; points[[idx[0], 1]]],
                matrix![points[[idx[1], 0]]; points[[idx[1], 1]]],
                matrix![points[[idx[2], 0]]; points[[idx[2], 1]]],
            )
        });
        let triangles = order_points(triangles);
        let (angles, triangles): (Vec<[F; 2]>, Vec<Triangle<F>>) = triangles
            .map(|t| (t.angles_sorted_discard(), t))
            .filter(|(angles, _)| angles.iter().all(|&a| a > self.min_angle))
            .unzip();

        let angles_flat = angles.into_iter().flatten().collect_vec();
        let hashes = Array2::from_shape_vec((angles_flat.len() / 2, 2), angles_flat).unwrap();

        (hashes, triangles)
    }

    fn get_transformation_matrix(
        xy1: Matrix3x2<F>,
        xy2: Matrix3x2<F>,
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
    use ndarray_rand::{
        rand_distr::{Normal, Uniform},
        RandomExt,
    };
    use numpy::{PyArray2, PyArray3, PyArrayMethods, ToPyArray};
    use pyo3::{ffi::c_str, prelude::*, types::PyTuple};
    use rand::prelude::*;

    #[test]
    fn order_points() {
        let mut rng = rand::rng();

        let triangles = (0..10)
            .map(|_| {
                Triangle::<f64>::new(
                    Vector2::new(rng.random(), rng.random()),
                    Vector2::new(rng.random(), rng.random()),
                    Vector2::new(rng.random(), rng.random()),
                )
            })
            .collect_vec();

        let ordered_points = super::order_points(triangles.iter().cloned())
            .collect_vec()
            .into_ndarray3();

        let ordered_points_py = Python::with_gil(|py| {
            let twirl_tri = py.import("twirl.triangles").unwrap();

            let arr = triangles.into_ndarray3().to_pyarray(py);
            let ordered_points_py = twirl_tri
                .call_method1("order_points", (arr,))
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .clone();
            ordered_points_py.to_owned_array()
        });

        assert_eq!(ordered_points, ordered_points_py);
    }

    #[test]
    fn hashes() {
        let points = Array2::random((20, 2), Uniform::new(0., 10.));

        let ast = TriangleAsterism::default();
        let (hashes, triangles) = ast.hashes(points.view());

        let (hashes_py, triangles_py) = Python::with_gil(|py| {
            let twirl_tri = py.import("twirl.triangles").unwrap();

            let arr = points.to_pyarray(py);
            let hashes_triangles_py = twirl_tri
                .call_method1("hashes", (arr,))
                .unwrap()
                .downcast::<PyTuple>()
                .unwrap()
                .clone();
            let hashes_py = hashes_triangles_py
                .get_item(0)
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap()
                .clone();
            let triangles_py = hashes_triangles_py
                .get_item(1)
                .unwrap()
                .downcast::<PyArray3<f64>>()
                .unwrap()
                .clone();

            (hashes_py.to_owned_array(), triangles_py.to_owned_array())
        });

        assert_abs_diff_eq!(hashes, hashes_py, epsilon = 1e-6);
        assert_eq!(triangles.into_ndarray3(), triangles_py);
    }

    #[test]
    fn find_matches() {
        let points1 = Array2::random((20, 2), Uniform::new(0., 1.));
        let points2 =
            points1.clone() + Array2::<f64>::random((20, 2), Normal::new(0., 0.007).unwrap());

        let ast = TriangleAsterism::default();
        let (hashes1, _) = ast.hashes(points1.view());
        let (hashes2, _) = ast.hashes(points2.view());

        let pairs = TriangleAsterism::find_matches(hashes1.clone(), hashes2.clone(), 0.02);

        let pairs_py = Python::with_gil(|py| {
            let fun: Py<PyAny> = PyModule::from_code(
                py,
                c_str!(
                    "from scipy.spatial import cKDTree

def pairs(hashes_pixels, hashes_radecs):
    tree_pixels = cKDTree(hashes_pixels)
    tree_radecs = cKDTree(hashes_radecs)
    pairs = []

    ball_query = tree_pixels.query_ball_tree(tree_radecs, r=0.02)

    for i, j in enumerate(ball_query):
        if len(j) > 0:
            pairs += [[i, k] for k in j]

    return pairs"
                ),
                c_str!("test"),
                c_str!("test"),
            )
            .unwrap()
            .getattr("pairs")
            .unwrap()
            .into();

            let hashes1_py = hashes1.to_pyarray(py);
            let hashes2_py = hashes2.to_pyarray(py);
            let pairs_py = fun.call1(py, (hashes1_py, hashes2_py)).unwrap();
            pairs_py.extract::<Vec<Vec<usize>>>(py).unwrap()
        });

        let mut pairs_sorted = pairs.into_iter().map(|a| a.to_vec()).collect_vec();
        pairs_sorted.sort_by_key(|x| x[0]);

        assert_eq!(pairs_sorted, pairs_py);
    }
}
