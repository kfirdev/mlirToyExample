#include "include/ToyLang/Dialect/Arrays/ArraysInterfaces.h"
#include "include/ToyLang/Dialect/Arrays/ArraysOps.h"

namespace mlir::toylang::arrays{
	void ConcatOp::inferShapes(){
  		auto finalType = ArrayType::get(
				getContext(),getLhs().getType().getLength()+getRhs().getType().getLength(),
				getLhs().getType().getType());
		getResult().setType(finalType);
	}
	void InsertOp::inferShapes(){
		getResult().setType(getDest().getType());
	}
}
