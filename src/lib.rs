#![warn(missing_docs)]

//! Rust port of the [astrometric plate solving package twirl](https://github.com/lgrcia/twirl). \
//! Currently, no source detection or catalog query is included. This only calculates the transformation
//! between a set of detected sources and a provided catalog. To that end, asterisms of three or four stars
//! are built, hashed, and then matched. \
//! Since it is faithful to the original, most documentation on the chosen algorithm applies.
//!
//! ## Interface
//! The central struct of this library is [`Twirs`]. It is used to choose the asterism scheme,
//! specify the sets of points and matching parameters, and find the plate solutions. \
//! In most cases, it should be created with [`Twirs::triangles()`] or [`Twirs::quads()`],
//! but for generic contexts, the asterism can be specified in [`Twirs::new()`].
//! In any case, additional parameters are set via `Twirs::with_*()` functions.
//!
//! Example:
//! ```rust
//! Twirs::quads(pixels, radecs)
//!     .with_tolerance(0.5)
//!     .with_hash_tolerance(1.5);
//! ```
//!
//! After constructing the struct, there are two options:
//! - Directly calculate the least squares matrix between the 1-padded matrix of pixel and sky coordinates.
//! - Calculate the [`Wcs`](wcs::Wcs) transformation.
//!
//! Both can also be executed in parallel.
//!
//! ## Parameters
//! - `hash_tolerance`: Maximum allowed Euclidean distance between hashes.
//!     This is used in the nearest-neighbor search for matches between pixel and sky coordinates.
//! - `tolerance`: Maximum allowed Euclidean distance between points after applying the transformation.
//!     This is used in counting the number of matches to check which transformation is the best.
//! - `min_match`: Minimum number of matches between the pixel and sky coordinates after applying the transformation.
//!     If this number of matches is reached, the search for transformations is stopped.
//!     A lower number might lead to faster runtimes, but might also terminate the program before finding the optimal transformation.\
//!     **Warning: If using parallel execution (e.g. [`Twirs::find_transform_par()`]), this option is ignored.**\
//!
//! Only when using triangle asterisms:
//! - `min_angle`: Minimum angle of triangle asterisms.
//!     Triangles not fulfilling this condition are excluded from the transformation search.

pub(crate) mod geometry;
pub(crate) mod matching;
pub(crate) mod ndarray_utils;
pub(crate) mod quads;
pub(crate) mod triangles;
pub mod wcs;

pub use matching::{Asterism, Twirs};
pub use quads::QuadAsterism;
pub use triangles::TriangleAsterism;

/// A generic float trait such that the plate solving algorithm is generic over `f32`/`f64`.
///
/// This trait is automatically implemented for all types implementing the supertraits.
/// Particularly, this includes `f32` and `f64`.
/// [`num_traits::Float`] is not a supertrait as the need to specify the provider of the redundant definitions of the basic math functions would clutter the code.
pub trait Float: Copy + Default + nalgebra::RealField + num_traits::FromPrimitive {}

impl<F> Float for F where F: Copy + Default + nalgebra::RealField + num_traits::FromPrimitive {}
