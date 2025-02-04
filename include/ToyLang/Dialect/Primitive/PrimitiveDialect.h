#pragma once

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace llvm{

template <typename T>
hash_code hash_value(SmallVector<T> S) {
  return hash_combine_range(S.begin(), S.end());
  //return hashing::detail::hash_combine_range_impl(S.begin(), S.end());
}

}

#include "mlir/IR/DialectImplementation.h"
#include "ToyLang/Dialect/Primitive/PrimitiveDialect.h.inc"
