#ifndef TENSOR_OPERANDS_H
#define TENSOR_OPERANDS_H

//#include "tensor_f_decl.h"

//#include <numeric>


//template <typename T, std::size_t N>
//Tensor<T, N> operator+ (const Tensor<T, N>& a, const Tensor<T, N>& b)
//{
//    Tensor<T, N> result = a;
//    result += b;
//    return result;
//}

//template <typename T, std::size_t N>
//Tensor<T, N> operator- (const Tensor<T, N>& a, const Tensor<T, N>& b)
//{
//    Tensor<T, N> result = a;
//    result -= b;
//    return result;
//}

//template <typename T>
//T operator* (const Tensor<T, 1>& a, const Tensor<T, 1>& b)
//{
//    assert(a.size() == b.size());
//    return std::inner_product(a.cbegin(), a.cend(), b.cbegin(), T{0});
//}

//template <typename T>
//T operator* (const Tensor_ref<T, 1>& a, const Tensor_ref<T, 1>& b)
//{
//    assert(a.size() == b.size());
//    return std::inner_product(a.cbegin(), a.cend(), b.cbegin(), T{0});
//}


//template <typename T>
//Tensor<T, 2> operator* (const Tensor<T, 2>& a, const Tensor<T, 2>& b)
//{
//    assert(a.cols() == b.rows());
//    auto r = a.rows(), c = b.cols();
////    Tensor<T, 2> result(r, c);



////    return result;
//}


#endif // TENSOR_OPERANDS_H
