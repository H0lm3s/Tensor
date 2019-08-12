#ifndef TENSOR_F_DECL_H
#define TENSOR_F_DECL_H

#include <iostream>

#include "../macros.h"

NUM_BEGIN

template <typename T, std::size_t N>
class Tensor;

template <typename T, std::size_t N>
class Tensor_ref;

template<std::size_t N>
struct Tensor_slice;

NUM_END

#endif // TENSOR_F_DECL_H
