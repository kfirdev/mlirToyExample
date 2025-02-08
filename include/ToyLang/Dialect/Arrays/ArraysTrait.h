#pragma once

#include "mlir/IR/OpDefinition.h"
#include "include/ToyLang/Dialect/Arrays/ArraysInterface.h"
#include "include/ToyLang/Dialect/Arrays/ArraysType.h"

namespace mlir::toylang::arrays {

template <typename ConcreteType>
class TwoXReusltLength : public OpTrait::TraitBase<ConcreteType, TwoXReusltLength> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
	  auto Lhs = mlir::dyn_cast<ArraysTypeInterface>(op->getOperandTypes()[0]);
	  auto Rhs = mlir::dyn_cast<ArraysTypeInterface>(op->getOperandTypes()[1]);
	  auto Reuslt = mlir::dyn_cast<ArraysTypeInterface>(op->getResultTypes()[0]);
	  op->getResults()[0].setType(Lhs.getCombined(Rhs.getLength()));

    return success();
  }
};

}
