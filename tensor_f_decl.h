#ifndef TENSOR_DECL_H
#define TENSOR_DECL_H

#include <iostream>


template <typename T, std::size_t N>
class Tensor;

template <typename T, std::size_t N>
class Tensor_ref;

template<std::size_t N>
struct Tensor_slice;


#endif // TENSOR_DECL_H
