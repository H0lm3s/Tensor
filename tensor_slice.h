#ifndef TENSOR_SLICE_H
#define TENSOR_SLICE_H

#include <iostream>
#include <array>
#include <cassert>

#include "tensor_traits.h"
#include "tensor_support.h"


/// Generic N-D tensor
template <std::size_t N>
struct Tensor_slice {

    /// ctors
    /**
     * @brief tensor_slice default ctor.
     */
    Tensor_slice()
        : start{0},
          size{0},
          extents{0},
          strides{0}
    {}

    /**
     * @brief tensor_slice ctor. Pass start point and extents
     * @param s
     * @param exts
     */
    Tensor_slice(size_t s,
                 std::initializer_list<std::size_t> exts)
        : start{s}
    {
        assert(exts.size() == N);
        std::copy(exts.begin(), exts.end(), extents.begin());
        size = tensor_impl::_calc_strides(extents, strides);
    }

    /**
     * @brief tensor_slice ctor. Pass start point,
     *        extents and strides.
     * @param s
     * @param exts
     * @param strs
     */
    Tensor_slice(size_t s,
                 std::initializer_list<std::size_t> exts,
                 std::initializer_list<std::size_t> strs)
        : start{s}
    {
        assert(exts.size() == N);
        std::copy(exts.begin(), exts.end(), extents.begin());
        std::copy(strs.begin(), strs.end(), strides.begin());
        size = std::accumulate(exts.begin(), exts.end(), 1,
                               std::multiplies<std::size_t> {});
    }

    /**
     * @brief tensor_slice variadic ctor that specify dimensions of tensor
     * @param dims
     */
    template <typename... Dims>
    Tensor_slice(Dims... dims)
        : start{0}
    {
        static_assert(sizeof...(Dims) == N,
                      "tensor_slice<N>::tensor_slice (Dims...): dimension mismatch");

        std::size_t args[N] { std::size_t(dims)... };
        std::copy(std::begin(args), std::end(args), extents.begin());
        size = tensor_impl::_calc_strides<N>(extents, strides);
    }

    /**
     * @brief operator () overload.
     * @param dims... parameter pack
     * @return the index of element calculated as the product
     *         between each dimesion offset with corresponding
     *         index in the N-Dimensional structure.
     *         It's defined only if all dimensions are size_t.
     *         Otherwise it is not.
     */
    template <typename... Dims,
             std::size_t NN = N,
             typename = Enable_if<tensor_impl::_requesting_element()>,
             typename = Enable_if<NN >= 3>>
    std::size_t
    operator()(Dims... dims) const
    {
        static_assert (sizeof... (Dims) == N,
                       "tensor_slice<N>::operator(): dimensions mismatch");

        size_t args[N] { size_t(dims)... };
        return start + std::inner_product(args, args + N, strides.begin(), size_t {0});
    }

    /**
     * @brief operator () overload. Applyed only in matrix.
     * @param i
     * @param j
     * @return the index of element
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 2>>
    std::size_t
    operator()(std::size_t i, std::size_t j) const
    { return i * strides[0] + j; }

    /**
     * @brief operator () overload. Applyed only in vector.
     * @param i
     * @param j
     * @return the index of element
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 1>>
    std::size_t
    operator()(std::size_t i) const
    { return i; }

    /**
     * @brief flat_index. As operator () but passing
     *        indexes in array
     * @param i
     * @return index of element in flat structure.
     */
    std::size_t
    flat_index(const std::array<std::size_t, N>& i) const
    {
        assert(i.size() == N);
        return start + std::inner_product(
                    i.begin(),
                    i.end(),
                    strides.begin(),
                    size_t {0});
    }

    /// Offset
    std::size_t start;

    /// Number of elements
    std::size_t size;

    /// Dimensions of tensor.
    std::array<size_t, N> extents;

    /// Strides of tensor in flat representation.
    std::array<size_t, N> strides;

};

/// Specialization to avoid inconsistent type
template <>
struct Tensor_slice<0>;

///-----------------------------------------------------------------------------------------------------------///
/// Overloading operators

template <std::size_t N>
inline bool operator==(const Tensor_slice<N> &a,
                       const Tensor_slice<N> &b)
{ return a.start == b.start &&
         std::equal(a.extents.cbegin(), a.extents.cend(), b.extents.cbegin()) &&
         std::equal(a.strides.cbegin(), a.strides.cend(), b.strides.cbegin()); }

template <std::size_t N>
inline bool operator!=(const Tensor_slice<N> &a,
                       const Tensor_slice<N> &b)
{ return !(a == b); }

///-----------------------------------------------------------------------------------------------------------///
/// Debug functions

template <std::size_t N>
std::ostream &operator<<(std::ostream &os, const std::array<std::size_t, N> &a) {
    for (auto e : a) os << e << " ";
    return os;
}

template <std::size_t N>
std::ostream &operator<<(std::ostream &os, const Tensor_slice<N> &s) {
    os << "size: " << s.size << "\nextents: "
       << s.extents << "\nstrides: "
       << s.strides << "\nstart: " << s.start;
    return os;
}

#endif // TENSOR_SLICE_H
