#ifndef TENSOR_ALIASES_H
#define TENSOR_ALIASES_H

#include "../macros.h"
#include "tensor_f_decl.h"

NUM_BEGIN

template <typename T>
using Cube = Tensor<T, 3>;

template <typename T>
using Mat = Tensor<T, 2>;

template <typename T>
using Vec = Tensor<T, 1>;

NUM_END

#endif // ALIASES_H
