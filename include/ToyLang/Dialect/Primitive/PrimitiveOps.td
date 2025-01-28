#ifndef LIB_DIALECT_PRIM_PRIMOPS_TD_
#define LIB_DIALECT_PRIM_PRIMOPS_TD_

include "PrimitiveDialect.td"
include "PrimitiveTypes.td"
include "PrimitiveAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PrimInt_AddOp : Op<Primitive_Dialect, "add", [Pure,SameOperandsAndResultType]> {
	let summary = "Addition operation between to ints";
	let arguments = (ins Prim_IntegerType:$lhs, Prim_IntegerType:$rhs);
	let results = (outs Prim_IntegerType:$output);
	let assemblyFormat =  "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
	let hasFolder = 1;
}

def PrimInt_ConstantOp : Op<Primitive_Dialect, "constant", [Pure, ConstantLike]> {
	let	summary = "Define a constnat int";
	let arguments = (ins Prim_IntegerAttr:$value);
	//let arguments = (ins AnyI32Attr:$value);
	let results = (outs Prim_IntegerType:$output);
	let assemblyFormat = "$value attr-dict `:` type($output)";

	let builders = [
		OpBuilder<(ins "Type":$type, "IntegerAttr":$value)>
	];
	let skipDefaultBuilders = 1;

	let hasVerifier = 1;
	let hasFolder = 1;
}

#endif
