#ifndef OPERANDS_H
#define OPERANDS_H

#include <numeric>
#include <cassert>

#include "tensor_f_decl.h"
#include "tensor_base.h"
#include "support.h"
#include "traits.h"

#include "../macros.h"

NUM_BEGIN

/// ------------------------------- EQUALITY - INEQUALITY ---------------------------- ///

/**
 * @brief operator ==. Equality
 * @param x
 * @param y
 */
template <typename T,
          typename = Enable_if<_tensor_type<T>()>>
inline
bool
operator==(const T& x, const T& y)
{
    assert(x.descriptor().extents == y.descriptor().extents);
    return std::equal(x.cbegin(), x.cend(), y.cbegin());
}

/**
 * @brief operator !=. Inequality
 * @param x
 * @param y
 */
template <typename T,
          typename = Enable_if<_tensor_type<T>()>>
inline
bool
operator!=(const T& x, const T& y)
{
    assert(x.descriptor().extents ==
           y.descriptor().extents);
    return !(x == y);
}

/// ------------------------------- SUM AND SUB WISE --------------------------------- ///

/**
 * @brief operator +.
 * @param a
 * @param b
 * @return Tensor
*/
template <typename T,
          typename = Enable_if<_tensor_type<T>()>>
Tensor<typename T::value_type, T::order>
operator+ (const T& a,
           const T& b)
{
    Tensor<typename T::value_type, T::order> result(a);
    result += b;
    return result;
}

/**
 * @brief operator -.
 * @param a
 * @param b
 * @return a new Tensor
 */
template <typename T,
          typename = Enable_if<_tensor_type<T>()>>
T
operator- (const T& a,
           const T& b)
{
    T result = a;
    result -= b;
    return result;
}

/// ------------------------------------- PRODUCT ------------------------------------ ///

/**
 * @brief operator *. Vec x Vec
 * @param a
 * @param b
 * @return value_type of the Tensor
 */
template <typename T1, typename T2,
          typename = Enable_if<(_1d<T1>() && _1d<T2>())>>
typename T1::value_type
operator* (const T1& a,
           const T2& b)
{
    assert(a.size() == b.size());
    return std::inner_product(a.cbegin(),
                              a.cend(),
                              b.cbegin(),
                              typename T1::value_type{0});
}

/**
 * @brief operator *. Vec x Mat
 * @param a
 * @param b
 * @return Vec
 */
template <typename T1, typename T2,
          typename = Enable_if<(_1d<T1>() && _2d<T2>())>>
Tensor<typename T1::value_type, 1>
operator* (const T1& a,
           const T2& b)
{
    assert(a.size() == b.rows());
    auto c = b.cols();

    Tensor<typename T1::value_type, 1> result(c);

    for (std::size_t i = 0; i < c; ++i)
        result(i) = a * b.col(i);

    return result;
}

/**
 * @brief operator *. Mat x Mat
 * @param a
 * @param b
 * @return Vec
 */
template <typename T1, typename T2,
          typename = Enable_if<(_2d<T1>() && _2d<T2>())>>
Tensor<typename T1::value_type, 2>
operator* (const T1& a,
           const T2& b)
{
    assert(a.cols() == b.rows());
    auto r = a.rows(), c = b.cols();

    Tensor<typename T1::value_type, 2> result(r, c);

    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            result(i, j) = a.row(i) * b.col(j);

    return result;
}

NUM_END

#endif // OPERANDS_H
