#ifndef TENSOR_OPERANDS_H
#define TENSOR_OPERANDS_H

#include <numeric>
#include <cassert>

#include "tensor_f_decl.h"

#include "../macros.h"

NUM_BEGIN


/**
 * @brief operator +. Sum two 2x2
 *        Tensor (matrix).
 * @param a
 * @param b
 * @return a new Tensor (matrix)
 */
template <typename T, std::size_t N>
Tensor<T, N>
operator+ (const Tensor<T, N>& a,
           const Tensor<T, N>& b)
{
    Tensor<T, N> result = a;
    result += b;
    return result;
}

/**
 * @brief operator -. Subtract two 2x2
 *        Tensor (matrix).
 * @param a
 * @param b
 * @return a new Tensor (matrix)
 */
template <typename T, std::size_t N>
Tensor<T, N>
operator- (const Tensor<T, N>& a,
           const Tensor<T, N>& b)
{
    Tensor<T, N> result = a;
    result -= b;
    return result;
}

/**
 * @brief operator +. Add two 2x2
 *        Tensor_ref (Matrix).
 * @param a
 * @param b
 * @return a new Tensor (matrix)
 */
template <typename T>
Tensor<T, 2>
operator+ (const Tensor_ref<T, 2>& a,
           const Tensor_ref<T, 2>& b)
{
    Tensor<T, 2> result(a);
    result += b;
    return result;
}

/**
 * @brief operator +. Subtract two 2x2
 *        Tensor_ref (matrix).
 * @param a
 * @param b
 * @return a new Tensor (matrix)
 */
template <typename T>
Tensor<T, 2>
operator- (const Tensor_ref<T, 2>& a,
           const Tensor_ref<T, 2>& b)
{
    Tensor<T, 2> result(a);
    result -= b;
    return result;
}

/**
 * @brief operator +. Add two
 *        Tensor_ref (vector).
 * @param a
 * @param b
 * @return a new Tensor (vector)
 */
template <typename T>
Tensor<T, 1>
operator+ (const Tensor_ref<T, 1>& a,
           const Tensor_ref<T, 1>& b)
{
    Tensor<T, 1> result(a);
    result += b;
    return result;
}

/**
 * @brief operator -. Subtract two
 *        Tensor_ref (vector).
 * @param a
 * @param b
 * @return a new Tensor (vector)
 */
template <typename T>
Tensor<T, 1>
operator- (const Tensor_ref<T, 1>& a,
           const Tensor_ref<T, 1>& b)
{
    Tensor<T, 1> result(a);
    result -= b;
    return result;
}

/**
 * @brief operator *. Dot product of 2
 *        Tensor (vector)
 * @param a
 * @param b
 * @return a T value.
 */
template <typename T>
T
operator* (const Tensor<T, 1>& a,
           const Tensor<T, 1>& b)
{
    assert(a.size() == b.size());
    return std::inner_product(a.cbegin(), a.cend(), b.cbegin(), T{0});
}

/**
 * @brief operator *. Dot product of 2
 *        Tensor_ref (vector)
 * @param a
 * @param b
 * @return a T value.
 */
template <typename T>
T
operator* (const Tensor_ref<T, 1>& a,
           const Tensor_ref<T, 1>& b)
{
    assert(a.size() == b.size());
    return std::inner_product(a.cbegin(),
                              a.cend(),
                              b.cbegin(),
                              T{0});
}

/**
 * @brief operator *. Multiply 2
 *        Tensor (matrix).
 * @param a
 * @param b
 * @return a matrix
 */
template <typename T>
Tensor<T, 2>
operator* (const Tensor<T, 2>& a,
           const Tensor<T, 2>& b)
{
    assert(a.cols() == b.rows());
    auto r = a.rows(), c = b.cols();
    Tensor<T, 2> result(r, c);

    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            result(i, j) = a.row(i) * b.col(j);

    return result;
}

/**
 * @brief operator *. Multiply 2
 *        Tensor_ref (matrix).
 * @param a
 * @param b
 * @return a matrix
 */
template <typename T>
Tensor<T, 2>
operator* (const Tensor_ref<T, 2>& a,
           const Tensor_ref<T, 2>& b)
{
    assert(a.cols() == b.rows());
    auto r = a.rows(), c = b.cols();
    Tensor<T, 2> result(r, c);

    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            result(i, j) = a.row(i) * b.col(j);

    return result;
}

NUM_END

#endif // TENSOR_OPERANDS_H
