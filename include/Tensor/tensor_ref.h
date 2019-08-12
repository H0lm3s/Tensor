#ifndef TENSOR_REF_H
#define TENSOR_REF_H

#include <iostream>
#include <vector>
#include <cassert>

#include "tensor_base.h"
#include "tensor_initializer.h"
#include "tensor_f_decl.h"

#include "../macros.h"

NUM_BEGIN


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
    Tensor_ref& operator=(const Tensor<U, N>& t)
    {
        assert(this->desc.extents == t.descriptor().extents);
        std::copy(t.cbegin(), t.cend(), begin());
        return *this;
    }

    /**
     * @brief operator ().
     * @param dims
     * @return the elements indexed by dims.
     */
    template<typename... Dims>
    Enable_if<tensor_impl::_requesting_element(), reference>
    operator()(Dims... dims)
    {
        static_assert (sizeof... (dims) == N,
                       "Tensor_ref<T, N>::operator(): dimension mismatch");
        assert(tensor_impl::_check_bounds(this->_desc, dims...));
        return *(_elems + this->_desc(dims...));
    }

    /**
     * @brief operator () (const).
     * @param dims
     * @return the elements indexed by dims.
     */
    template<typename... Dims>
    Enable_if<tensor_impl::_requesting_element(), const_reference>
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
     * @brief operator []. Only in a vector, get i-th value.
     * @param i
     * @return const_reference.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 1>>
    const T&
    operator[] (std::size_t i) const
    { return this->operator()(i); }

    /**
     * @brief operator []. Only in a vector, get i-th value.
     * @param i
     * @return reference.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 1>>
    T&
    operator[] (std::size_t i)
    { return this->operator()(i); }

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
     * @brief operator =. Assign b to a.
     * @param value
     * @return *this
     */
    Tensor_ref&
    operator= (const T& value)
    { return apply([&](T& a) { a = value; }); }

    /**
     * @brief operator +=. Sum a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor_ref&
    operator+= (const T& value)
    { return apply([&](T& a) { a += value; }); }

    /**
     * @brief operator -=. Subtract a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor_ref&
    operator-= (const T& value)
    { return apply([&](T& a) { a -= value; }); }

    /**
     * @brief operator *=. Multiplicate a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor_ref&
    operator*= (const T& value)
    { return apply([&](T& a) { a *= value; }); }

    /**
     * @brief operator /=. Divide a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor_ref&
    operator/= (const T& value)
    { return apply([&](T& a) { a /= value; }); }

    /**
     * @brief operator %=. Module a and b and put
     *        in a. Just for integers.
     * @param value
     * @return *this
     */
    Tensor_ref&
    operator%= (const T& value)
    { return apply([&](T& a) { a %= value; }); }

    /**
     * @brief apply. For each element, apply the
     *        predicate f, using values of another
     *        tensor.
     * @param f
     * @param value
     * @return *this.
     */
    template <typename F, typename M>
    Enable_if<_tensor_type<M>(), Tensor_ref&>
    apply(F f, M& m)
    {
        assert(this->_desc.extents == m.descriptor().extents);
        for (auto i = begin(), j = m.cbegin(); i != end(); ++i, ++j)
            f(*i, *j);
        return *this;
    }

    /**
     * @brief operator +=. Add tensor b to a
     *        and put result to a.
     * @param b
     * @return this.
     */
    template <typename M>
    Enable_if<_tensor_type<M>(), Tensor_ref&>
    operator+= (const M& t)
    { return apply([&](T &a,
                   const typename M::value_type& b) { a += b; }, t); }

    /**
     * @brief operator -=. Subtract tensor b to a
     *        and put result to a.
     * @param b
     * @return this.
     */
    template <typename M>
    Enable_if<_tensor_type<M>(), Tensor_ref&>
    operator-= (const M& t)
    { return apply([&](T& a,
                   const typename M::value_type& b) { a -= b; }, t); }

    /**
     * @brief data.
     * @return a pointer to the first element
     */
    T*
    data()
    { return _elems; }

    /**
     * @brief data. (const)
     * @return a pointer to the first element
     */
    T*
    data() const
    { return _elems; }

    /// Iterators
    /**
     * @brief begin.
     * @return iterator pointing to begin position.
     */
    iterator begin()
    { return {_elems, this->_desc}; }

    /**
     * @brief end.
     * @return iterator pointing to an element
     *         after end position.
     */
    iterator end()
    { return {_elems, this->_desc, true}; }

    /**
     * @brief begin.
     * @return const_iterator pointing to begin position.
     */
    const_iterator cbegin() const
    { return {_elems, this->_desc}; }

    /**
     * @brief end.
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
 *        a input iterator.
 */
template<typename T, size_t N>
class Tensor_iterator {

public:

    /// Aliases
    using value_type = typename std::remove_const<T>::type;
    using iterator_category = std::input_iterator_tag; /// THIS MUST BE CHANGED
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;
    using difference_type = std::ptrdiff_t;

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

///-----------------------------------------------------------------------------------------------------------///
/// Debug functions

template <typename T>
std::ostream &operator<<(std::ostream& os, const Tensor_ref<T, 2>& t) {
    os << "{\n";
    for (std::size_t i = 0; i < t.rows(); ++i) {
        os << " { ";
        for (std::size_t j = 0; j < t.cols() - 1; ++j)
            os << t(i, j) << ", ";
        os << t(i, t.cols() - 1) << " }\n";
    }
    os << "}";
    return os;
}

template <typename T>
std::ostream &operator<<(std::ostream& os, const Tensor_ref<T, 1>& t) {
    os << "{ ";
    for (std::size_t i = 0; i < t.size() - 1; ++i)
        os << t(i) << ", ";
    os << t(t.size() - 1) << " }";
    return os;
}

NUM_END

#endif // TENSOR_REF_H
