# Tensor
A simple library written in C++, to make operation with a N-dimensional tensor. Since this library is in development, it's not complete. Some operations are defined only for 1-2 dimensional structures.


## Overview

I wrote this library when I was reading 'The C++ Programming Language (4th Edition, Stroustrup)'. It provides:
  + Constructors and Assignments.
  + Initialization similiar to std::vector.
  + Access to element by operators and method.
  + Scalar Operations.
  + Some Matrix and Vector operations.
  + (in dev) Iterator compatible (up to now, range-for, std::copy, ...)

### Members (public)

   - *begin()* = iterator pointing to the first element.
   - *end()* = iterator pointing to the next element after the last.

   - *cbegin()* = as begin() but const.
   - *cend()* = as end() but const.

   - *data()* = get a pointers list of elements. (as std::vector).

   - *apply(F f)* = apply a predicate to all elements.

   - *slice<D>(std::size_t offset)* = D is the dimension, offset is the number of the substructure.

   - *size()* = number of elements.
   - *extents(std::size_t n)* = get the number of elements in n-th dimension.

   - *descriptor()* = get the object descriptor, which contains all informations of the structure.
   - *rows()* = only in 2d Tensor, return number of rows. Is the same of extents(0).
   - *cols()* = only in 2d Tensor, return number of cols. Is the same of extents(1).
   - *row(std::size_t n)* = only in 2d Tensor, return n-th row. Is the same of slice<0>(n).
   - *col(std::size_t n)* = only in 2d Tensor, return n-th col. Is the same of slice<1>(n).
   - *order* = get the number of dimensions.

 
## Include on your project 
1. Clone the repository.
   ```sh
   git clone https://github.com/s4mu313/Tensor.git
   ```
2. Include the directory in your project.
3. Include tensor.h as shown below.

## How to use
```
#include <iostream>
#include "include/tensor.h"

int main()
{

    /// Constructors with any type
    /// Vector
    Math::Tensor<std::string, 1> v { "1", "2", "3", "4" };

    /// Matrix
    Math::Tensor<int, 2> m1 { {1, 2, 3},
                              {4, 5, 6},
                              {7, 8, 9} };

    /// Matrix
    Math::Tensor<int, 2> m2 { {10, 11, 12},
                              {13, 14, 15},
                              {16, 17, 18} };

    /// Cube
    Math::Tensor<double, 3> c1 { { {1.0, 2.0}, {3.0, 4.0} },
                                 { {5.0, 6.0}, {7.0, 8.0} } };



    /// Assignments  /// m1 = { {10, 11, 12},
    m1 = m2;         ///        {13, 14, 15},
                     ///        {16, 17, 18} }

    /// Access

    m1(1, 1) = 500;  /// m1 = { {10, 11, 12},
                     ///        {13, 500, 15},
                     ///        {16, 17, 18} }


    c1(0, 0, 0) = 9.0;     /// c1 { { {1.7, 2.0}, {3.0, 4.0} },
                           ///      { {5.0, 6.0}, {7.0, 8.0} } }


    auto mat_ref_1 = c1.slice<0>(0); /// Do a slice on the first dimension
                                     /// to get a reference on the elements
                                     /// of the cube, containing a matrix
                                     /// with: { { 9.0, 2.0 },
                                     ///         { 3.0, 4.0 } }
                                     /// The template parameter in <> is the
                                     /// dimension, the function parameter ()
                                     /// is the offset to get the desired
                                     /// structure with N - 1 dimensions
                                     /// Pay attention! By using auto you
                                     /// get a reference to the elements
                                     /// of cube. If you need a copy
                                     /// declare a new Tensor with correct
                                     /// dimensions.

    Math::Tensor<double, 2> mat_ref_2 = c1.slice<1>(0);

    /// row and col works only in 2-dimensional Tensor.
    auto r = m1.row(0);   /// Reference to row_0 of m1. r is a vector with {1, 2, 3}
    r = m1[0];       /// Same as above.
    auto c = m1.col(0);   /// Reference to col_0 of m1. c is vector {1, 4, 7}

    r = c; /// r and c are both reference of m1 row_0 and col_0, so
           /// change their value, means change m1.
           /// If you need a copy you have to declare a new tensor
           /// as shown below:

    Math::Tensor<double, 1> r_copy = m1.row(0);

    /// Operations

    /// Scalar

    m2 += 3;     /// m2 = { {13, 14, 15},
                 ///        {16, 17, 18},
                 ///        {19, 20, 21} }

    m2 -= 1;     /// Sub 1 to all elements

    m2 *= 2;     /// Mul 2 to all elements

    m2 /= 10;    /// Div by 10 all elements

    /// Basic Vector-Matrix operations

    Math::Tensor<int, 2> sum = m1 + m2; /// Sum is a new matrix

    auto sum2 = m1 + m2; /// Same as above
    auto sub = m2 - m1;

    m2 += m1; /// Sum and put to m2
    m2 -= m1; /// Subtract and put to m2


    Math::Tensor<double, 1> v1 { 1.0, 2.0 };

    Math::Tensor<double, 2> m3 { {1.0, 2.0},
                                 {3.0, 4.0} };

    Math::Tensor<double, 2> m4 { {5.0, 6.0},
                                 {7.0, 8.0} };


    /// Vector-Matrix product
    Math::Tensor<double, 1> prod = v1 * m3;  /// { 7, 10 }

    /// Matrix-Matrix product
    Math::Tensor<double, 2> prod3 = m3 * m4; /// { {19.0, 22.0},
                                             ///   {43.0, 50.0} };

    /// Iterable

    for(auto it = m1.begin(); it != m1.end(); ++it) // it++ it's also defined
        std::cout << *it << std::endl;

    for(const auto& x : m1.row(0))
        std::cout << x << std::endl;


    /// TIPS: there are some "pre-build" types:
    /// Vec is declared as Tensor<T, 1>.
    /// Mat is declared as Tensor<T, 2>.
    /// Cube is declared as Tensor<T, 3>.
    /// H_Cube is declared as Tensor<T, 4>.

    Math::Vec<double> vec { 1, 2, 3, 4 };

    Math::Mat<double> mat { {1, 2},
                            {3, 4} };

    Math::Cube<double> cube { { {1.0, 2.0}, {3.0, 4.0} },
                              { {5.0, 6.0}, {7.0, 8.0} } };

    Math::H_Cube<double> h_cube { { { {1.0, 2.0}, {1.0, 2.0}}, { {3.0, 4.0}, {3.0, 4.0 } } },
                                  { { {1.0, 2.0}, {1.0, 2.0}}, { {3.0, 4.0}, {3.0, 4.0 } } } };

    
    /// Apply a predicate to all elements
    mat.apply([](double& d){d += 500;}); /// using a lambda

    struct op { void operator()(double& d) { d += 500; } };
    mat.apply(op()); /// or using a functor



    return 0;
}

```

