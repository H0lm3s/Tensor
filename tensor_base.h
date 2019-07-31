#ifndef Tensor_base_H
#define Tensor_base_H

#include <iostream>
#include <vector>
#include <cassert>

#include "tensor_slice.h"


/*
 * This class is the "base" form of Tensor and Tensor_ref.
 * It contains common elements for both of them.
*/
template <typename T, std::size_t N>
class Tensor_base {
public:

    static constexpr std::size_t order = N;
    using value_type = T;

    /// Default base ctors.
    Tensor_base() = default;
    Tensor_base(Tensor_base &&) = default;
    Tensor_base &operator=(Tensor_base &&) = default;
    Tensor_base(const Tensor_base &) = default;
    Tensor_base &operator=(const Tensor_base &) = default;
    ~Tensor_base() = default;

    /**
     * @brief Tensor_base ctor. Create from desc.
     * @param desc
     */
    Tensor_base(const Tensor_slice<N>& desc)
        : _desc{desc}
    {}

    /**
     * @brief Tensor_base. Create a tensor by passing dimensions
     *                     and initialize it to default value.
     * @param exts
     */
    template<typename... Exts>
    explicit Tensor_base(Exts... exts) :
        _desc{exts...}
    {}

    /**
     * @brief extent. Get the i-th dimension
     * @param i
     * @return the i-th dimension
     */
    std::size_t
    extent(std::size_t i) const
    {
        assert(i < order);
        return this->_desc.extents[i];
    }

    /**
     * @brief size.
     * @return the number of elements in tensor.
     */
    std::size_t
    size() const
    { return _desc.size; }

    /**
     * @brief descriptor. Get descriptor.
     * @return the descriptor.
     */
    const Tensor_slice<N>&
    descriptor() const
    { return this->_desc; }

    /**
     * @brief rows. Get rows number only if
     *              it is a matrix.
     * @return number of rows in the tensor.
     */
    template<std::size_t NN = N,
             typename = Enable_if<NN == 2>>
    std::size_t
    rows() const
    { return this->_desc.extents[0]; }

    /**
     * @brief cols. Get cols number only if
     *              it is a matrix.
     * @return number of cols in the tensor.
     */
    template<std::size_t NN = N,
             typename = Enable_if<NN == 2>>
    std::size_t
    cols() const
    { return this->_desc.extents[1]; }

protected:
    Tensor_slice<N> _desc;

};

/// Debug function
template <typename T, std::size_t N>
std::ostream
&operator<<(std::ostream &os,
            const Tensor_base<T, N> &t) {
    os << t.descriptor() << "\n"
       << "\nOrder:  " << t.order << "\n";
    return os;
}


#endif // Tensor_base_H
