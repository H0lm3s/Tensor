#ifndef SUPPORT_H
#define SUPPORT_H

#include <iostream>
#include <numeric>
#include <array>
#include <cassert>

#include "tensor_f_decl.h"
#include "tensor_traits.h"


namespace tensor_impl {

/**
 * @brief _add_list. Recurring call to get the
 *        deepest initializer_list to get all
 *        value in each one and put them in to
 *        a vector. When the value is not
 *        initializer_list insert them in a
 *        vector.
 * @param first
 * @param last
 * @param v
 */
template <typename T, typename V>
void _add_list(const T* first, const T* last, V& v)
{ v.insert(v.end(), first, last); }

/**
 * @brief _add_list. Recurring call to get the
 *        deepest initializer_list to get all
 *        value in each one and put them in
 *        a vector.
 * @param first
 * @param last
 * @param v
 */
template <typename T, typename V>
void _add_list(const std::initializer_list<T>* first,
               const std::initializer_list<T>* last, V& v)
{
    for (; first != last; ++first)
        _add_list(first->begin(), first->end(), v);
}

/**
 * @brief _insert_flat. Insert all value in an
 *        initializer_list in a "flat" structure.
 * @param l
 * @param v
 */
template <typename T, typename V>
void _insert_flat(std::initializer_list<T> l, V& v)
{ _add_list(l.begin(), l.end(), v); }

/// ...just defined...implemented below
/**
 * @brief _check_non_jagged. Check if the
 *        tensor initializer is jagged.
 * @param l
 * @return true if it is, false otherwise.
 */
template <std::size_t N, typename L>
bool
_check_non_jagged(const L& l);

/**
 * @brief _add_extents.
 * @param first
 * @param l
 * @return
 */
template <std::size_t N, typename I, typename L>
Enable_if<N == 1> _add_extents(I& first, const L& l)
{ *first = l.size(); }

/**
 * @brief _add_extents. Insert size of each
 *        dimension to an array
 * @param first
 * @param l
 */
template <std::size_t N, typename I, typename L>
Enable_if<N >= 2, void> _add_extents(I& first, const L& l)
{
    assert(_check_non_jagged<N>(l));
    *first++ = l.size();
    _add_extents<N - 1>(first, *l.begin());
}

/**
 * @brief _derive_extents. Get an array with
 *        extents.
 * @param l
 * @return std::array with extents
 */
template <std::size_t N, typename L>
std::array<std::size_t, N>
_derive_extents(const L& l)
{
    std::array<std::size_t, N> a;
    auto f = a.begin();
    _add_extents<N>(f, l);
    return a;
}

/**
 * @brief _check_non_jagged. Check if the
 *        tensor initializer is jagged.
 * @param l
 * @return true if it is, false otherwise.
 */
template <std::size_t N, typename L>
bool
_check_non_jagged(const L& l)
{
    auto i = l.begin();
    for (auto j = i + 1; j != l.end(); ++j)
        if (_derive_extents<N - 1>(*i) !=
                _derive_extents<N - 1>(*j))
            return false;
    return true;
}

/**
 * @brief calc_strides. Calculate the offset
 *        between elements for each dimension
 *        in flat representiation
 * @param exts
 * @param strs
 * @return the number of element in the structure.
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
