#ifndef TENSOR_INITIALIZER_H
#define TENSOR_INITIALIZER_H

#include <iostream>


/**
 * @brief The tensor_init struct. Recursive struct that contains nested initializer_list
 *        from N to 1
 */
template <typename T, std::size_t N>
struct Tensor_init
{ using type = std::initializer_list<typename Tensor_init<T, N - 1>::type>; };

template <typename T>
struct Tensor_init<T, 1>
{ using type = std::initializer_list<T>; };

/// Specialization to avoid inconsistences
template <typename T>
struct Tensor_init<T, 0>;

template <typename T, size_t N>
using Tensor_initializer = typename Tensor_init<T, N>::type;

#endif // TENSOR_INITIALIZER_H
