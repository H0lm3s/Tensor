#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

#include "tensor_base.h"
#include "tensor_ref.h"
#include "tensor_initializer.h"


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
    Tensor(Tensor_initializer<T, N> &t_init) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    ~Tensor() = default;

    /// Ctors and assignement.
    template <typename U>
    Tensor(const Tensor_ref<U, N>& t_ref)
    {
    }

    template <typename U>
    Tensor& operator= (const Tensor_ref<U, N>& t_ref)
    {
    }

    template <typename... Exts>
    explicit Tensor(Exts... exts)
    {
    }

    Tensor(Tensor_initializer<T, N> t_init)
    {
    }

    Tensor& operator=(Tensor_initializer<T, N> init)
    {
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

    /// Access to elements
    template <typename... Args>
    Enable_if<tensor_impl::_requesting_element<Args...>(), T&>
    operator()(Args... args)
    {
    }

    /// Access to elements (const)
    template <typename... Args>
    Enable_if<tensor_impl::_requesting_element<Args...>(), const T&>
    operator()(Args... args)
    {
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
        return {this->_desc, t_slice};
    }

    /**
     * @brief row. Only in a matrix, return a row slice.
     * @param i
     * @return Tensor.
     */
    template <std::size_t NN = N,
              typename = Enable_if<NN == 2>>
    Tensor_ref<const T, N - 1>
    row(std::size_t i) const
    {
        Tensor_slice<N - 1> t_slice;
        tensor_impl::_slice_dim<0>(i, this->_desc, t_slice);
        return {this->_desc, t_slice};
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
        return {this->_desc, t_slice};
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
        return {this->_desc, t_slice};
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
     * @brief apply. Apply the predicate f
     *        to all elements in the tensor.
     * @param f
     * @return *this.
     */
    template <typename F>
    Tensor& apply(F f)
    {
        for (auto& x : _elems)
            f(x);
        return *this;
    }

    /**
     * @brief apply. For each element, apply the
     *        predicate f, by passing a value to it.
     * @param f
     * @param value
     * @return *this.
     */
    template <typename F>
    Tensor& apply(F f, const T& value)
    {
        for (auto& x : _elems)
            f(x, value);
        return *this;
    }

    /**
     * @brief apply. For each element, apply the
     *        predicate f, by passing another tensor.
     * @param f
     * @param value
     * @return *this.
     */
//    template <typename F, typename M>
//    Enable_if<Tensor_type<M>, Tensor&>
//    Tensor& apply(M& m, F f)
//    {
//    }

    /**
     * @brief operator =. Assign b to a.
     * @param value
     * @return *this
     */
    Tensor&
    operator= (const T& value)
    {
        this->apply(tensor_impl::assign<T>(), value);
        return *this;
    }

    /**
     * @brief operator +=. Sum a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator+= (const T& value)
    {
        this->apply(tensor_impl::sum<T>(), value);
        return *this;
    }

    /**
     * @brief operator -=. Subtract a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator-= (const T& value)
    {
        this->apply(tensor_impl::sub<T>(), value);
        return *this;
    }

    /**
     * @brief operator *=. Multiplicate a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator*= (const T& value)
    {
        this->apply(tensor_impl::mul<T>(), value);
        return *this;
    }

    /**
     * @brief operator /=. Divide a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator/= (const T& value)
    {
        this->apply(tensor_impl::div<T>(), value);
        return *this;
    }

    /**
     * @brief operator %=. Module a and b and put in a.
     * @param value
     * @return *this
     */
    Tensor&
    operator%= (const T& value)
    {
        this->apply(tensor_impl::mod<T>(), value);
        return *this;
    }

//    template <typename M>
//    Enable_if<Tensor_type<M>, Tensor&>
//    operator+= (const M& b)

//    template <typename M>
//    Enable_if<Tensor_type<M>, Tensor&>
//    operator-= (const M& b)

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

#endif // TENSOR_H
