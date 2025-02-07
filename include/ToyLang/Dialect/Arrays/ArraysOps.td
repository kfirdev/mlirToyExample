#ifndef LIB_DIALECT_ARR_ARROPS_TD_
#define LIB_DIALECT_ARR_ARROPS_TD_

include "ArraysDialect.td"
include "ArraysAttr.td"
include "ArraysType.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AttrInt_ConstantOp : Op<Arrays_Dialect, "constant", [Pure, ConstantLike]> {
	let	summary = "Define a constnat int";
	//let arguments = (ins AnyIntElementsAttr:$value);
	let arguments = (ins AnyAttr:$value);
	let results = (outs AnyType:$output);
	//let assemblyFormat = "$value attr-dict `:` qualified(type($output))";
	let hasCustomAssemblyFormat = 1;

	//let builders = [
		//OpBuilder<(ins "Type":$type, "IntegerArrType":$value)>
	//];
	//let skipDefaultBuilders = 1;

	//let hasVerifier = 1;
	//let hasFolder = 1;
}

#endif
