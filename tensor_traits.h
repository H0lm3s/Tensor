#ifndef TRAITS_H
#define TRAITS_H

#include <iostream>

/**
 * @brief Convertible. Check if X is convertible to Y
 * @return true if it is convertible, false otherwise.
 */
template <typename X, typename Y>
constexpr bool
Convertible()
{ return std::is_convertible<X, Y>::value; }

/**
 * @brief All.
 * @return true if all args... are true, false otherwise.
 */
constexpr bool
All()
{ return true; }

template <typename... Args>
constexpr bool
All(bool b, Args... args)
{ return b && All(args...); }

/**
 * @brief Some.
 * @param b
 * @param args
 * @return true if at least one element is true, false otherwise.
 */
constexpr bool
Some()
{ return false; }

template <typename... Args>
constexpr bool
Some(bool b, Args... args)
{ return b || Some(args...); }

///----------------------------------------------------------------------------///

template <bool B, typename T = void>
using Enable_if = typename std::enable_if<B, T>::type;

#endif // TRAITS_H
