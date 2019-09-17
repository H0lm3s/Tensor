#ifndef TRAITS_H
#define TRAITS_H

#include <iostream>

#include "tensor_f_decl.h"

#include "../macros.h"

NUM_BEGIN


/// Alias of std::enable_if...
template <bool B, typename T = void>
using Enable_if = typename std::enable_if<B, T>::type;

/// Element to create a "false" case.
struct _failure
{};

/// Element to represent "true".
template <typename T>
struct _success : std::true_type
{};

/// Element to represent "false".
template <>
struct _success<_failure> : std::false_type
{};

/// Concepts
template <typename M>
struct _get_type {

    template <typename T, std::size_t N, typename = Enable_if<N >= 1>>
    static _success<void> check (const Tensor<T, N>& t);

    template <typename T, std::size_t N, typename = Enable_if<N >= 1>>
    static _success<void> check (const Tensor_ref<T, N>& t);

    static _failure check(...);

    using type = decltype (check(std::declval<M>()));
};

/// Struct to check value type T.
template <typename T>
struct _has_tensor_type : _success<typename _get_type<T>::type>
{};

/**
 * @brief _tensor_type. Check if T is a Tensor.
 * @return true if it is, false otherwise.
 */
template <typename T>
constexpr bool
_tensor_type()
{ return _has_tensor_type<T>::value; }

/**
 * @brief Convertible. Check if X is convertible to Y
 * @return true if it is convertible, false otherwise.
 */
template <typename X, typename Y>
constexpr bool
Convertible()
{ return std::is_convertible<X, Y>::value; }

/**
 * @brief All.
 * @return true if all args... are true, false otherwise.
 */
constexpr bool
All()
{ return true; }

template <typename... Args>
constexpr bool
All(bool b, Args... args)
{ return b && All(args...); }

/// CHECKING THE DIMENSION OF A TENSOR/TENSOR_REF

/// Concepts
template <typename M>
struct _1d_type {

    template <typename T>
    static _success<void> check (const Tensor<T, 1>& t);

    template <typename T>
    static _success<void> check (const Tensor_ref<T, 1>& t);

    static _failure check(...);

    using type = decltype (check(std::declval<M>()));
};

/// Concepts
template <typename M>
struct _2d_type {

    template <typename T>
    static _success<void> check (const Tensor<T, 2>& t);

    template <typename T>
    static _success<void> check (const Tensor_ref<T, 2>& t);

    static _failure check(...);

    using type = decltype (check(std::declval<M>()));
};

/// Struct to check value type T.
template <typename T>
struct _1d: _success<typename _1d_type<T>::type>
{};

/// Struct to check value type T.
template <typename T>
struct _2d: _success<typename _2d_type<T>::type>
{};

/**
 * @brief _is_1d. Check if T has just 1 dimension.
 * @return true if it is, false otherwise.
 */
template <typename T>
constexpr bool
_is_1d()
{ return _1d<T>::value; }

/**
 * @brief _is_2d. Check if T has just 2 dimensions.
 * @return true if it is, false otherwise.
 */
template <typename T>
constexpr bool
_is_2d()
{ return _2d<T>::value; }


NUM_END

#endif // TRAITS_H
