#pragma once
#include "mlir/IR/OpDefinition.h"

namespace mlir::toylang::primitive {

template <typename ConcreteType>
class  IsAnInteger: public OpTrait::TraitBase<ConcreteType, IsAnInteger> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    return success();
  }
};
template <typename ConcreteType>
class  IsAFloat: public OpTrait::TraitBase<ConcreteType, IsAFloat> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    return success();
  }
};
template <typename ConcreteType>
class  IsABool: public OpTrait::TraitBase<ConcreteType, IsABool> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    return success();
  }
};
template <typename ConcreteType>
class  IsAString: public OpTrait::TraitBase<ConcreteType, IsAString> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    return success();
  }
};

}
