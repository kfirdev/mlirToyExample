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

}
