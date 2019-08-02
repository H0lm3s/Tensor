#ifndef SUPPORT_H
#define SUPPORT_H

#include <iostream>
#include <numeric>
#include <array>

#include "tensor_f_decl.h"
#include "tensor_traits.h"


namespace tensor_impl {

/**
 * Functor to perform assignement, addition, subtraction, multiplication, division and module
 */

/// Assign.
template <typename T>
struct assign { void operator() (T& a, T& b) { a = b; } };

/// Sum.
template <typename T>
struct sum { void operator() (T& a, T& b) { a += b; } };

/// Sub.
template <typename T>
struct sub { void operator() (T& a, T& b) { a -= b; } };

/// Mul.
template <typename T>
struct mul { void operator() (T& a, T& b) { a *= b; } };

/// Div.
template <typename T>
struct div { void operator() (T& a, T& b) { a /= b; } };

/// Mod.
template <typename T>
struct mod { void operator() (T& a, T& b) { a %= b; } };

///----------------------------------------------------------------------------///

/**
 * @brief calc_strides. Calculate the offset between elements for each dimension
 *                      in flat representiation
 * @param exts
 * @param strs
 * @return the number of element in the structure
 */
template <size_t N>
std::size_t
_calc_strides(const std::array<std::size_t, N>& exts,
              std::array<std::size_t, N>& strs)
{
    std::size_t st = 1;
    for (auto i = N; i > 0; --i) {
        strs[i - 1] = st;
        st *= exts[i - 1];
    }
    return st;
}

/**
 * @brief _calc_size. Calculate number of elements
 *        in the N-dimensional structure.
 * @param exts
 * @return size of N-dimensional structure.
 */
template <typename V>
std::size_t
_calc_size(const V& exts)
{ return std::accumulate(exts.begin(), exts.end(), 1,
                         std::multiplies<std::size_t>{}); }

/**
 * @brief _slice_dim. Calculate the new descriptor of the
 *        N-dimensional structure, from a descriptor,
 *        applying an offset (to indexing a specific slice).
 * @param offset
 * @param src
 * @param dst
 */
template <std::size_t D, std::size_t N>
void
_slice_dim(std::size_t offset, const Tensor_slice<N>& src, Tensor_slice<N - 1>& dst)
{

    static_assert (N > 0,
                   "_slice_dim<D>: error N must be greater than 0");
    dst.start = src.start;
    std::size_t j = N - 2;

    for (std::size_t i = N; i > 0; --i)
        if (i - 1 == D)
            dst.start += src.strides[i - 1] * offset;
        else {
            dst.extents[j] = src.extents[i - 1];
            dst.strides[j] = src.strides[i - 1];
            --j;
        }

    dst.size = _calc_size(dst.extents);
}

/**
 * @brief _check_bounds. Checks indexes passed are
 *        lower than dimension of the structure.
 * @param ts
 * @param dims
 * @return true if they are, false otherwise.
 */
template <std::size_t N, typename... Dims>
bool _check_bounds(const Tensor_slice<N>& ts, Dims... dims)
{
    std::size_t indexes[N] {std::size_t(dims)...};
    return std::equal(indexes,
                      indexes + N,
                      ts.extents.begin(),
                      std::less<std::size_t>{});
}

/**
 * @brief _requesting_element. Checks that all args
 *        are size_t
 * @return true if they are, false otherwise.
 */
template <typename... Args>
constexpr bool _requesting_element()
{ return All((Convertible<Args, std::size_t>())...); }

};



#endif // SUPPORT_H
