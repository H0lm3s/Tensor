#ifndef Tensor_ref_H
#define Tensor_ref_H

#include <iostream>
#include <vector>
#include <cassert>

#include "tensor_base.h"
#include "tensor_initializer.h"
#include "tensor_f_decl.h"


template <typename T, std::size_t N>
class Tensor_iterator;

/**
 * @brief The Tensor_ref class. Contains a reference to
 *        all values stored in N-D tensor and provide access
 *        to them.
 */
template <typename T, std::size_t N>
class Tensor_ref : public Tensor_base<T, N> {

public:

    /// Aliases
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;
    using value_type = T;
    using iterator = Tensor_iterator<T, N>;
    using const_iterator = const Tensor_iterator<T, N>;

    /// Deleted ctor.
    Tensor_ref() = delete;
    Tensor_ref& operator=(Tensor_init<T, N>&) = delete;

    /// Default ctor.
    Tensor_ref(Tensor_ref&&) = default;
    Tensor_ref(const Tensor_ref&) = default;
    Tensor_ref& operator=(Tensor_ref&) = default;
    Tensor_ref& operator=(const Tensor_ref&) = default;
    ~Tensor_ref() = default;

    /// ctor with param
    Tensor_ref(const Tensor_slice<N>& desc, pointer elems)
        : Tensor_base<T, N>{desc},
          _elems{elems}
    {}

    /// ctor. Create Tensor_ref from Tensor.
    template <typename U>
    Tensor_ref(Tensor<U, N>& t)
    {
        assert(this->desc.extents == t.descriptor().extents);
        std::copy(t.begin(), t.end(), begin());
        return *this;
    }

    /**
     * @brief operator ().
     * @param dims
     * @return the elements indexed by dims.
     */
    template<typename... Dims>
    Enable_if<tensor_impl::_requesting_element(), reference>
    operator()(Dims... dims) const
    {
        static_assert (sizeof... (dims) == N,
                       "Tensor_ref<T, N>::operator(): dimension mismatch");
        assert(tensor_impl::_check_bounds(this->_desc, dims...));
        return *(_elems + this->_desc(dims...));
    }

    /**
     * @brief slice. Get a slice of a N-dimensional
     *        structure by a specific dimension.
     *        Create a new descriptor and return a
     *        Tensor_ref with N - 1 dimensions.
     * @param i
     * @return Tensor_ref<T, N- 1>.
     */
    template<std::size_t D>
    Tensor_ref<T, N - 1>
    slice(std::size_t i) const
    {
        static_assert (D < N, "Tensor_ref<T, N - 1>::Dimension of slice "
                              "(D) must be lower than N");

        assert(i < this->_desc.extents[D]);
        Tensor_slice<N - 1> t;
        tensor_impl::_slice_dim<D>(i, this->_desc, t);
        return {t, _elems};
    }

    /**
     * @brief row. Only in a matrix, return a row slice.
     * @param i
     * @return Tensor_ref.
     */
    template<std::size_t NN = N,
             typename = Enable_if<NN == 2>>
    Tensor_ref<T, N - 1>
    row(std::size_t i) const
    { return slice<0>(i); }

    /**
     * @brief col. Only in a matrix, return a col slice.
     * @param i
     * @return Tensor_ref.
     */
    template<std::size_t NN = N,
             typename = Enable_if<NN == 2>>
    Tensor_ref<T, N - 1>
    col(std::size_t i) const
    { return slice<1>(i); }

    /**
     * @brief operator []. Only in a matrix, make row slice.
     * @param i
     * @return Tensor_ref.
     */
    template<std::size_t NN = N,
             typename = Enable_if<NN == 2>>
    std::size_t
    operator[](std::size_t i) const
    { return row(i); }

    /**
     * @brief apply. Apply the predicate F
     *        to all value of N-dimensional structure.
     *        Applyied to all specializations.
     * @param f
     * @return this.
     */
    template<typename F>
    Tensor_ref<T, N>&
    apply(F f)
    {
        for (auto x = begin(); x != end(); ++x)
            f(*x);
        return *this;
    }

    /**
     * @brief data.
     * @return a pointer to the first element
     */
    T*
    data()
    { return _elems; }

    /// Iterators
    /**
     * @brief begin.
     * @return iterator pointing to begin position.
     */
    iterator begin() const
    { return {_elems, this->_desc}; }

    /**
     * @brief end.
     * @return iterator pointing to an element
     *         after end position.
     */
    iterator end() const
    { return {_elems, this->_desc, true}; }

    /**
     * @brief cbegin.
     * @return const_iterator pointing to begin position.
     */
    const_iterator cbegin() const
    { return {_elems, this->_desc}; }

    /**
     * @brief cend.
     * @return const_iterator pointing to an
     *         element after end position.
     */
    const_iterator cend() const
    { return {_elems, this->_desc, true}; }


private:
    T* _elems;
};

/// Specialization to avoid inconsistent type.
template <typename T>
class Tensor_ref<T, 0>;


/**
 * @brief The Tensor_iterator class. It's like
 *        a fwd iterator.
 */
template<typename T, size_t N>
class Tensor_iterator {

    /// Aliases
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

public:

    /// Ctor.
    Tensor_iterator(T* t, const Tensor_slice<N>& s, bool end = false)
        : _desc{s}, _data_ptr{t}
    {
        std::fill(_pos.begin(), _pos.end(), 0);
        if (end) {
            _pos[0] = _desc.extents[0];
            _data_ptr = t + _desc.flat_index(_pos);
        } else
            _data_ptr = t + _desc.start;
    }

    /**
     * @brief operator ++. Make sequentially increment
     * @return *this
     */
    Tensor_iterator& operator++() {
        for (std::size_t i = N; i != 0; --i) {
            if (++_pos[i - 1] < _desc.extents[i - 1])
                return *this;
            _pos[i - 1] = 0;
        }
        _pos[0] = _desc.extents[0];
        return *this;
    }

    /**
     * @brief operator *. To get the pointed-to.
     * @return a reference to the element.
     */
    reference
    operator* ()
    { return *(_data_ptr + _desc.flat_index(_pos)); }

    /**
     * @brief operator !=.
     * @param t
     * @return true if they are the same or false otherwise.
     */
    bool
    operator!= (const Tensor_iterator<T, N>& t) const
    { return _pos != t._pos; }

private:

    /// Element indexes
    std::array<std::size_t, N> _pos;

    /// Reference to descriptor
    const Tensor_slice<N>& _desc;

    /// Pointer to flat data
    T* _data_ptr;

};

///-----------------------------------------------------------------------------------------------------------///
/// Overloading operators

template <typename T, std::size_t N>
inline bool operator==(const Tensor_iterator<T, N> &a,
                       const Tensor_iterator<T, N> &b)
{
    assert(a.descriptor() == b.descriptor());
    return &*a == &*b;
}

template <typename T, std::size_t N>
inline bool operator!=(const Tensor_iterator<T, N> &a,
                       const Tensor_iterator<T, N> &b)
{ return !(a == b); }


#endif // Tensor_ref_H
