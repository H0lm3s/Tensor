#ifndef TRAITS_H
#define TRAITS_H

#include <iostream>

#include "tensor_f_decl.h"


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

/**
 * @brief Some.
 * @param b
 * @param args
 * @return true if at least one element is true, false otherwise.
 */
constexpr bool
Some()
{ return false; }

template <typename... Args>
constexpr bool
Some(bool b, Args... args)
{ return b || Some(args...); }


#endif // TRAITS_H
