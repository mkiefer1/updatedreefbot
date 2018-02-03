// All rights reserved.
// Carnegie Mellon University
//
// Author: Mark Desnoyer
// Date: Nov 21, 2007
//
// hash_map is not in the standard stl, so depending on your compiler
// it will be in a different spot with a different namespace. This
// file will try to deal withat that complexity under the hood.
//
// TODO(mdesnoyer): make this platform independent. It currently only
// works for gcc 3+
#ifndef _BASE_HASH_MAP_H
#define _BASE_HASH_MAP_H

#include <ext/hash_map>
using __gnu_cxx::hash_map;

#endif  // _BASE_HASH_MAP_H
