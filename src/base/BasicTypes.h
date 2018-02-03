// Copyright (c) 2007, Carnegie Mellon University
// All rights reserved.
//
// Author: Mark Desnoyer
// Date: Oct 21, 2007
//
// Defines all the system types in such a way that they are platform
// independent.
//
// Some sections adapted from open source Google code.
// Copyright 2004-2005, Google Inc.

#ifndef _BASE_BASICTYPES_H
#define _BASE_BASICTYPES_H

#include <inttypes.h>

// Explicitly define the integer types
typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;

// Same thing for floating point types
typedef float float32;
typedef double float64;

// Defines NULL because it's not in the C++ standard
#ifdef NULL
#undef NULL
#endif
#define NULL 0

// A macro to disallow the evil copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_EVIL_CONSTRUCTORS(TypeName)    \
  TypeName(const TypeName&);                    \
  void operator=(const TypeName&);

using namespace std;

// Common functions to return the max or min of two values.
// Enforces type checking.
//template<typename T>
//const T& MAX(const T& a, const T& b) { return a > b ? a : b; }
//template<typename T>
//const T& MIN(const T& a, const T& b) { return a < b ? a : b; }

#endif  // _BASE_BASICTYPES_H
