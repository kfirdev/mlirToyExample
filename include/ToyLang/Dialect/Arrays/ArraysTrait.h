#pragma once

#include "mlir/IR/OpDefinition.h"
#include "include/ToyLang/Dialect/Arrays/ArraysInterface.h"
#include "include/ToyLang/Dialect/Arrays/ArraysType.h"

namespace mlir::toylang::arrays {

template <typename ConcreteType>
class TwoXReusltLength : public OpTrait::TraitBase<ConcreteType, TwoXReusltLength> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
	  auto Lhs = mlir::dyn_cast<ArrayType>(op->getOperandTypes()[0]);
	  auto Rhs = mlir::dyn_cast<ArrayType>(op->getOperandTypes()[1]);
	  auto Reuslt = mlir::dyn_cast<ArrayType>(op->getResultTypes()[0]);
	  auto new_type = ArrayType::get(op->getContext(),Lhs.getLength()+Rhs.getLength(),Lhs.getType());
	  op->getResults()[0].setType(new_type);

    return success();
  }
};

}
