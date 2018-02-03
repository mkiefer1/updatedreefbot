// Copyright (c) 2007, Carnegie Mellon University
// All rights reserved.
//
// Author: Mark Desnoyer
// Date: Oct 21, 2007
//
// The standard libraries don't include a hash function for the string
// class. This header provides it so that string can be used in
// hash_map, hash_set etc.
//
// TODO(mdesnoyer) Make this platform independent
#ifndef _BASE_STRINGHASH_H
#define _BASE_STRINGHASH_H

#include <hash_fun.h>

namespace __gnu_cxx {

template<>
struct hash<std::string> {
  std::size_t operator()(const std::string& s) const {
    return hasher_(s.c_str());
  }
  hash<const char*> hasher_;
};

template<>
struct hash<const std::string> {
  std::size_t operator()(const std::string& s) const {
    return hasher_(s.c_str());
  }
  hash<const char*> hasher_;
};

}  // namespace

#endif  // _BASE_STRINGHASH_H
