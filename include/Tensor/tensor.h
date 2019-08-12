#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

#include "tensor_base.h"
#include "tensor_initializer.h"
#include "tensor_ref.h"

#include "../macros.h"

NUM_BEGIN


template <typename T, std::size_t N>
class Tensor : public Tensor_base<T, N> {
public:

    /// Aliases.
    using value_type = T;
    using reference = T&;
    using const_reference = const T&;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    /// Default ctors.
    Tensor() = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor() = default;

    /// Ctor from Tensor_ref
    template <typename U>
    Tensor(const Tensor_ref<U, N>& t_ref)
        : Tensor_base<T, N> (t_ref.descriptor().extents),
          _elems(t_ref.cbegin(), t_ref.cend())
    { static_assert (Convertible<U, T>(),
                     "Tensor constructor: types mismatch"); }

    /// Assignement from Tensor_ref
    template <typename U>
    Tensor& operator= (const Tensor_ref<U, N>& t_ref)
    {
        this->_desc = t_ref.descriptor();
        _elems.assign(t_ref.cbegin(), t_ref.cend());
        return *this;
    }

    /// Ctor by passing extents
    template <typename... Exts>
    explicit Tensor(Exts... exts)
        : Tensor_base<T, N> (exts...),
          _elems(this->_desc.size)
    {}

    /// Ctor from Tensor_initializer
    Tensor(Tensor_initializer<T, N> t_init)
    {
        this->_desc.extents = tensor_impl::_derive_extents<N>(t_init);
        this->_desc.size = tensor_impl::_calc_strides(this->_desc.extents,
                                                      this->_desc.strides);
        _elems.reserve(this->_desc.size);
        tensor_impl::_insert_flat(t_init, _elems);
        assert(_elems.size() == this->_desc.size);
    }

    /// Assignement ctor from Tensor_initializer
    Tensor& operator=(Tensor_initializer<T, N> t_init)
    {
        this->_desc.extents = tensor_impl::_derive_extents<N>(t_init);
        this->_desc.size = tensor_impl::_calc_strides(this->_desc.extents,
                                                      this->_desc.strides);
        _elems.clear();
        _elems.reserve(this->_desc.size);
        tensor_impl::_insert_flat(t_init, _elems);

        assert(_elems.size() == this->_desc.size);
        return *this;
    }

    /**
      * This is not active is N > 1,
      * beacuse it need Tensor_init,
      * but if it has just one dimension,
      * using a tensor_init or a std::initializer_list
      * does not make difference.
      */
    template <typename U,
              std::size_t NN = N,
              typename = Enable_if<(NN > 1)>,
    typename = Enable_if<Convertible<U, std::size_t>()>>
    Tensor(std::initializer_list<U>) = delete;

    /**
      * This is not active is N > 1,
      * beacuse it need Tensor_init,
      * but if it has just one dimension,
      * using a tensor_init or a std::initializer_list
      * does not make difference.
      */
    template <typename U,
              std::size_t NN = N,
              typename = Enable_if<(NN > 1)>,
    typename = Enable_if<Convertible<U, std::size_t>()>>
    Tensor& operator= (std::initializer_list<U>) = delete;

    /**
     * @brief data.
     * @return a pointer to the first element.
     */
    T*
    data()
    { return _elems.data(); }

    /**
     * @brief data.
     * @return a pointer to the first element.
     */
    const T*
    data() const
    { return _elems.data(); }

    /**
     * @brief slice. Get a slice of a N-dimensional
     *        structure by a specific dimension.
     *        Create a new descriptor and return a
     *        Tensor_ref with N - 1 dimensions.
     * @param i
     * @return Tensor_ref<T, N- 1> (const).
     */
    template<std::size_t D>
    const Tensor_ref<T, N - 1>
    slice(std::size_t i) const
    {
        static_assert (D < N, "Tensor_ref<T, N - 1>::Dimension of slice "
                              "(D) must be lower than N");

        assert(i < this->_desc.extents[D]);
        Tensor_slice<N - 1> t;
        tensor_impl::_slice_dim<D>(i, this->_desc, t);
        return {t, _elems.data()};
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
    slice(std::size_t i)
    {
        static_assert (D < N, "Tensor_ref<T, N - 1>::Dimension of slice "
                              "(D) must be lower than N");

        assert(i < this->_desc.extents[D]);
        Tensor_slice<N - 1> t;
        tensor_impl::_slice_dim<D>(i, this->_desc, t);
        return {t, _elems.data()};
    }

    /// Access to elements
    template <typename... Args>
    Enable_if<tensor_impl::_requesting_element<Args...>(), T&>
    operator()(Args... args)
    {
        assert(tensor_impl::_check_bounds(this->_desc, args...));
        return *(_elems.data() + this->_desc(args...));
    }

    /// Access to elements (const)
    template <typename... Args>
    Enable_if<tensor_impl::_requesting_element<Args...>(), const T&>
    operator()(Args... args) const
    {
        assert(tensor_impl::_check_bounds(this->_desc, args...));
        return *(_elems.data() + this->_desc(args...));
    }

    /**
     * @brief row. Only in a matrix, return a row slice.
     * @param i
     * @return Tensor.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 2>>
    Tensor_ref<T, N - 1>
    row(std::size_t i)
    {
        Tensor_slice<N - 1> t_slice;
        tensor_impl::_slice_dim<0>(i, this->_desc, t_slice);
        return {t_slice, _elems.data()};
    }

    /**
     * @brief row. Only in a matrix, return a row slice.
     * @param i
     * @return Tensor.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 2>>
    const Tensor_ref<const T, N - 1>
    row(std::size_t i) const
    {
        Tensor_slice<N - 1> t_slice;
        tensor_impl::_slice_dim<0>(i, this->_desc, t_slice);
        return {t_slice, _elems.data()};
    }

    /**
     * @brief col. Only in a matrix, return a col slice.
     * @param i
     * @return Tensor.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 2>>
    Tensor_ref<T, N - 1>
    col(std::size_t i)
    {
        Tensor_slice<N - 1> t_slice;
        tensor_impl::_slice_dim<1>(i, this->_desc, t_slice);
        return {t_slice, _elems.data()};
    }

    /**
     * @brief col. Only in a matrix, return a col slice.
     * @param i
     * @return Tensor.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 2>>
    Tensor_ref<const T, N - 1>
    col(std::size_t i) const
    {
        Tensor_slice<N - 1> t_slice;
        tensor_impl::_slice_dim<1>(i, this->_desc, t_slice);
        return {t_slice, _elems.data()};
    }

    /**
     * @brief operator []. Only in a matrix, make row slice.
     * @param i
     * @return Tensor.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 2>>
    Tensor_ref<T, N - 1>
    operator[] (std::size_t i)
    { return row(i); }

    /**
     * @brief operator []. Only in a matrix, make row slice.
     * @param i
     * @return Tensor.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 2>>
    Tensor_ref<const T, N - 1>
    operator[] (std::size_t i) const
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
     * @brief apply. Apply the predicate f
     *        to all elements in the tensor.
     * @param f
     * @return *this.
     */
    template <typename F>
    Tensor& apply(F f)
    {
        for (auto& x : _elems) f(x);
        return *this;
    }

    /**
     * @brief operator =. Assign b to a.
     * @param value
     * @return *this
     */
    Tensor&
    operator= (const T& value)
    { return apply([&](T& a) { a = value; }); }

    /**
     * @brief operator +=. Sum a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator+= (const T& value)
    { return apply([&](T& a) { a += value; }); }

    /**
     * @brief operator -=. Subtract a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator-= (const T& value)
    { return apply([&](T& a) { a -= value; }); }

    /**
     * @brief operator *=. Multiplicate a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator*= (const T& value)
    { return apply([&](T& a) { a *= value; }); }

    /**
     * @brief operator /=. Divide a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator/= (const T& value)
    { return apply([&](T& a) { a /= value; }); }

    /**
     * @brief operator %=. Module a and b and put
     *        in a. Just for integers.
     * @param value
     * @return *this
     */
    Tensor&
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
    Enable_if<_tensor_type<M>, Tensor&>
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
    Enable_if<_tensor_type<M>, Tensor&>
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
    Enable_if<_tensor_type<M>(), Tensor&>
    operator-= (const M& t)
    { return apply([&](T& a,
                   const typename M::value_type& b) { a -= b; }, t); }

    /**
     * @brief begin.
     * @return iterator pointing to begin position.
     */
    iterator begin()
    { return _elems.begin(); }

    /**
     * @brief end.
     * @return iterator pointing to an element
     *         after end position.
     */
    iterator end()
    { return _elems.end(); }

    /**
     * @brief cbegin.
     * @return const_iterator pointing to begin position.
     */
    const_iterator cbegin() const
    { return _elems.cbegin(); }

    /**
     * @brief cend.
     * @return const_iterator pointing to an element
     *         after end position.
     */
    const_iterator cend() const
    { return _elems.cend(); }

private:
    /// Elements
    std::vector<T> _elems;

};

///-----------------------------------------------------------------------------------------------------------///
/// Debug functions

template <typename T>
std::ostream &operator<<(std::ostream& os, const Tensor<T, 2>& t) {
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
std::ostream &operator<<(std::ostream& os, const Tensor<T, 1>& t) {
    os << "{ ";
    for (std::size_t i = 0; i < t.size() - 1; ++i)
        os << t(i) << ", ";
    os << t(t.size() - 1) << " }";
    return os;
}

NUM_END

#endif // TENSOR_H
