#![allow(non_snake_case)]
use nalgebra::allocator::Reallocator;
use nalgebra::{DefaultAllocator, Dim, DimAdd, DimSum, OMatrix, Rotation2, Vector2, U1};
use ndarray::{s, Array1, Array2};

use crate::Float;

pub(crate) fn pad<F: Float, R: Dim, C: Dim>(x: OMatrix<F, R, C>) -> OMatrix<F, R, DimSum<C, U1>>
where
    C: DimAdd<U1>,
    DefaultAllocator: Reallocator<F, R, C, R, DimSum<C, U1>>,
{
    let cols = x.shape().1;
    x.insert_column(cols, F::from_f64(1.).unwrap())
}

pub(crate) fn transform_points<F: Float>(
    mut points: Array2<F>,
    transformations_matrix: Array2<F>,
) -> Array2<F> {
    let n_points = points.shape()[0];
    points
        .push_column(Array1::from_elem(n_points, F::from_f64(1.).unwrap()).view())
        .unwrap();
    let mul = transformations_matrix.dot(&points.t());
    let slice = mul.slice(s![0..2, ..]);
    slice.t().to_owned()
}

/// Project `point` on a segment from `origin` to `axis` and sum over it.
pub(crate) fn projection_sum<F: Float>(
    point: &Vector2<F>,
    origin: &Vector2<F>,
    axis: &Vector2<F>,
) -> F {
    let mut n = axis - origin;
    n /= n.norm();
    (point - origin).component_mul(&n).sum()
}

/// Rotate `point` around `pivot` by `angle`.
pub(crate) fn rotation<F: Float>(point: &Vector2<F>, pivot: &Vector2<F>, angle: F) -> Vector2<F> {
    let R = Rotation2::new(angle);
    let X = R * (point - pivot);
    X + pivot
}

/// `(x, y)` basis as defined in Lang 2009.
pub(crate) fn u1u2<F: Float>(a: &Vector2<F>, b: &Vector2<F>) -> (Vector2<F>, Vector2<F>) {
    let x = rotation(b, a, -F::frac_pi_4());
    let y = rotation(b, a, F::frac_pi_4());
    (x, y)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::matrix;
    use numpy::{PyArray2, PyArrayMethods, ToPyArray};
    use pyo3::prelude::*;
    use rand::Rng;

    use crate::{matching::Asterism, quads::QuadAsterism};

    #[test]
    fn pad() {
        let mat = matrix![
            4., 3., 2.;
            7., 6., 5.;
            -3., -2., -1.
        ];
        let pad = matrix![
            4., 3., 2., 1.;
            7., 6., 5., 1.;
            -3., -2., -1., 1.
        ];

        assert_eq!(super::pad(mat), pad);
    }

    #[test]
    fn get_transformation_matrix() {
        let mut rng = rand::thread_rng();

        let quad1: nalgebra::Matrix<
            f64,
            nalgebra::Const<4>,
            nalgebra::Const<2>,
            nalgebra::ArrayStorage<f64, 4, 2>,
        > = matrix![
            rng.gen(), rng.gen();
            rng.gen(), rng.gen();
            rng.gen(), rng.gen();
            rng.gen(), rng.gen()
        ];
        let quad2: nalgebra::Matrix<
            f64,
            nalgebra::Const<4>,
            nalgebra::Const<2>,
            nalgebra::ArrayStorage<f64, 4, 2>,
        > = matrix![
            rng.gen(), rng.gen();
            rng.gen(), rng.gen();
            rng.gen(), rng.gen();
            rng.gen(), rng.gen()
        ];

        let lstsq = QuadAsterism::get_transformation_matrix(quad1, quad2).unwrap();

        let lstsq_py = Python::with_gil(|py| {
            let twirl_geo = py.import("twirl.geometry").unwrap();

            let point1_py = quad1.to_pyarray(py);
            let point2_py = quad2.to_pyarray(py);

            let lstsq_py = twirl_geo
                .call_method1("get_transform_matrix", (point1_py, point2_py))
                .unwrap()
                .downcast::<PyArray2<f64>>()
                .unwrap()
                .clone();

            //py.run_bound("del twirl.geometry", None, None).unwrap();

            lstsq_py.readonly().as_matrix().clone_owned()
        });

        assert_abs_diff_eq!(lstsq.as_slice(), lstsq_py.as_slice(), epsilon = 1e-6);
    }
}
