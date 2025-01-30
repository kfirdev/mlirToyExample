#ifndef LIB_DIALECT_PRIM_PRIMOPS_TD_
#define LIB_DIALECT_PRIM_PRIMOPS_TD_

include "PrimitiveDialect.td"
include "PrimitiveTypes.td"
include "PrimitiveAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Primitive_BinOpBase<string name> : Op<Primitive_Dialect, name, [Pure,SameOperandsAndResultType]>{
	let assemblyFormat =  "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
	let hasFolder = 1;
}

class Primitve_IntOpBase<string name> : Primitive_BinOpBase<name>{
	let arguments = (ins Prim_IntegerType:$lhs, Prim_IntegerType:$rhs);
	let results = (outs Prim_IntegerType:$output);
}


def PrimInt_AddOp : Primitve_IntOpBase<"add"> {
	let summary = "Addition operation between to ints";
}
def PrimInt_MultOp : Primitve_IntOpBase<"mul"> {
	let summary = "Addition operation between to ints";
}
def PrimInt_SubOp : Primitve_IntOpBase<"sub"> {
	let summary = "Addition operation between to ints";
}
def PrimInt_DivOp : Primitve_IntOpBase<"div"> {
	let summary = "Addition operation between to ints";
}

def PrimInt_ConstantOp : Op<Primitive_Dialect, "constant", [Pure, ConstantLike]> {
	let	summary = "Define a constnat int";
	//let arguments = (ins Prim_IntegerAttr:$value);
	let arguments = (ins AnyAttrOf<[Prim_IntegerAttr]>:$value);
	//let arguments = (ins AnyI32Attr:$value);
	let results = (outs Prim_IntegerType:$output);
	//let assemblyFormat = "$value attr-dict type($output)";
	//let assemblyFormat = "$value attr-dict `:` type($output)";
	let hasCustomAssemblyFormat = 1;

	let builders = [
		OpBuilder<(ins "Type":$type, "IntegerAttr":$value)>
	];
	let skipDefaultBuilders = 1;

	let hasVerifier = 1;
	let hasFolder = 1;
}

#endif
