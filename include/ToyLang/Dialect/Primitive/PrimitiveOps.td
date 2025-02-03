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
	let hasVerifier = 1;
}

class Primitve_OpBase<string name> : Primitive_BinOpBase<name>{
	let arguments = (ins PrimitiveTypeInterface:$lhs, PrimitiveTypeInterface:$rhs);
	let results = (outs PrimitiveTypeInterface:$output);
}


def Prim_AddOp : Primitve_OpBase<"add"> {
	let summary = "Addition operation between to ints";
}
def Prim_MultOp : Primitve_OpBase<"mul"> {
	let summary = "Addition operation between to ints";
}
def Prim_SubOp : Primitve_OpBase<"sub"> {
	let summary = "Addition operation between to ints";
}
def Prim_DivOp : Primitve_OpBase<"div"> {
	let summary = "Addition operation between to ints";
}

def PrimInt_ConstantOp : Op<Primitive_Dialect, "constant", [Pure, ConstantLike]> {
	let	summary = "Define a constnat int";
	//let arguments = (ins Prim_IntegerAttr:$value);
	let arguments = (ins PrimitiveAttrInterface:$value);
	//let arguments = (ins AnyI32Attr:$value);
	let results = (outs PrimitiveTypeInterface:$output);
	//let assemblyFormat = "$value attr-dict type($output)";
	//let assemblyFormat = "$value attr-dict `:` type($output)";
	let hasCustomAssemblyFormat = 1;

	let builders = [
		OpBuilder<(ins "Type":$type, "PrimitiveAttrInterface":$value)>
	];
	let skipDefaultBuilders = 1;

	let hasVerifier = 1;
	let hasFolder = 1;
}

#endif
