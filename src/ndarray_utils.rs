//! A collection of various utilities needed in this library.
//! The two main groups are `ndarray` to `nalgebra` conversions,
//! inspired by [`nshare`](https://github.com/rust-cv/nshare);
//! and common operations on arrays.

use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Matrix, Scalar, Storage};
use ndarray::{
    Array, Array2, Array3, ArrayView, ArrayView1, Axis, Dimension, RemoveAxis, ShapeBuilder,
};

use crate::quads::Quad;
use crate::triangles::Triangle;
use crate::Float;

pub trait IntoNdarray2 {
    type Out;

    fn into_ndarray2(self) -> Self::Out;
}

impl<N: Scalar> IntoNdarray2 for Array2<N> {
    type Out = Array2<N>;

    fn into_ndarray2(self) -> Self::Out {
        self
    }
}

impl<N: Scalar, C: Dim, R: Dim, S: Storage<N, R, C>> IntoNdarray2 for Matrix<N, R, C, S>
where
    DefaultAllocator: Allocator<R, C, Buffer<N> = S>,
{
    type Out = Array2<N>;

    fn into_ndarray2(self) -> Self::Out {
        Array2::from_shape_vec(
            self.shape().strides(self.strides()),
            self.into_iter().cloned().collect(),
        )
        .unwrap()
    }
}

pub trait IntoNdarray3 {
    type Out;

    fn into_ndarray3(self) -> Self::Out;
}

impl<N: Scalar> IntoNdarray3 for Array3<N> {
    type Out = Array3<N>;

    fn into_ndarray3(self) -> Self::Out {
        self
    }
}

impl<F: Float> IntoNdarray3 for Vec<Triangle<F>> {
    type Out = Array3<F>;

    fn into_ndarray3(self) -> Self::Out {
        Array3::from_shape_vec(
            (self.len(), 3, 2),
            self.into_iter()
                .flatten()
                .flat_map(|v| v.data.0.into_iter().flatten())
                .collect_vec(),
        )
        .unwrap()
    }
}

impl<F: Float> IntoNdarray3 for Vec<Quad<F>> {
    type Out = Array3<F>;

    fn into_ndarray3(self) -> Self::Out {
        Array3::from_shape_vec(
            (self.len(), 4, 2),
            self.into_iter()
                .flatten()
                .flat_map(|v| v.data.0.into_iter().flatten())
                .collect_vec(),
        )
        .unwrap()
    }
}

pub(crate) fn norm_axis<F, D, Di>(arr: ArrayView<F, Di>, axis: Axis) -> Array<F, D>
where
    F: Float,
    D: Dimension,
    Di: RemoveAxis<Smaller = D>,
{
    let norm_sq = arr.map(|x| (*x * *x)).sum_axis(axis);
    norm_sq.map(|x| x.sqrt())
}

pub(crate) fn min_axis<F, D, Di>(arr: ArrayView<'_, F, Di>, axis: Axis) -> Array<F, D>
where
    F: Float,
    D: Dimension,
    Di: RemoveAxis<Smaller = D>,
{
    arr.map_axis(axis, |a| {
        *a.into_iter()
            .min_by(|f1, f2| f1.partial_cmp(f2).unwrap())
            .unwrap()
    })
}

pub(crate) fn argmin<F: Float>(arr: ArrayView1<F>) -> usize {
    arr.iter()
        .enumerate()
        .min_by(|(_, value0), (_, value1)| value0.partial_cmp(value1).expect("found nan"))
        .expect("empty iterator")
        .0
}

pub(crate) fn max_axis<F, D, Di>(arr: ArrayView<'_, F, Di>, axis: Axis) -> Array<F, D>
where
    F: Float,
    D: Dimension,
    Di: RemoveAxis<Smaller = D>,
{
    arr.map_axis(axis, |a| {
        *a.into_iter()
            .max_by(|f1, f2| f1.partial_cmp(f2).unwrap())
            .unwrap()
    })
}

pub(crate) fn argmax<F>(arr: ArrayView1<'_, F>) -> usize
where
    F: Float,
{
    arr.indexed_iter()
        .reduce(|acc, f| if acc.1 >= f.1 { acc } else { f })
        .unwrap()
        .0
}

#[cfg(test)]
mod tests {
    use nalgebra::matrix;
    use ndarray::array;

    use super::*;
    use crate::triangles::Triangle;

    #[test]
    fn matrix_to_array2() {
        let matrix = matrix![1., 2., 3.; 4., 5., 6.; 7., 8., 9.];
        let arr = matrix.into_ndarray2();

        assert_eq!(arr, array![[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
    }

    #[test]
    fn triangles_to_array3() {
        let triangles: Vec<Triangle<f32>> = vec![
            Triangle::new(matrix![1.; 2.], matrix![3.; 4.], matrix![5.; 6.]),
            Triangle::new(matrix![7.; 8.], matrix![9.; 10.], matrix![11.; 12.]),
            Triangle::new(matrix![13.; 14.], matrix![15.; 16.], matrix![17.; 18.]),
        ];
        let arr = triangles.into_ndarray3();
        dbg!(&arr);
        assert_eq!(arr.shape(), &[3, 3, 2])
    }

    #[test]
    fn axis_norm() {
        let arr = array![[0., 1.], [2., 3.]];
        dbg!(super::norm_axis(arr.view(), Axis(1)));
    }

    #[test]
    fn axis_min() {
        let arr = array![[0., 1.], [2., 3.]];
        dbg!(super::min_axis(arr.view(), Axis(0)));
        dbg!(super::min_axis(arr.view(), Axis(1)));
    }

    #[test]
    fn axis_max() {
        let arr = array![[0., 1.], [2., 3.]];
        dbg!(super::max_axis(arr.view(), Axis(0)));
        dbg!(super::max_axis(arr.view(), Axis(1)));
    }
}
