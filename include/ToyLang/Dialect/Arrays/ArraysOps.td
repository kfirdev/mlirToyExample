#ifndef LIB_DIALECT_ARR_ARROPS_TD_
#define LIB_DIALECT_ARR_ARROPS_TD_

include "ArraysDialect.td"
include "ArraysAttr.td"
include "ArraysType.td"
include "../Primitive/PrimitiveInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TwoXReusltLength:  NativeOpTrait<"TwoXReusltLength"> {
  let cppNamespace = "::mlir::toylang::arrays";
}

class Array_ArrOpBase<string name, list<Trait> traits = [Pure]> : Op<Arrays_Dialect, name, traits>{
	let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
	let arguments = (ins ArraysTypeInterface:$lhs, ArraysTypeInterface:$rhs);
	let results = (outs ArraysTypeInterface:$output);
	//let hasFolder = 1;
	//let hasVerifier = 1;
}

def Arr_ConcatOp : Array_ArrOpBase<"concat",[Pure,TwoXReusltLength]> {
	let summary = "concat operation between to arrays";
	let hasVerifier = 1;
}

def Arr_ExtractOp : Array_ArrOpBase<"extract"> {
	let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor) `->` type($output)";
	let arguments = (ins Builtin_Index:$indices, ArraysTypeInterface:$tensor);
	let results = (outs PrimitiveTypeInterface:$output);
	let summary = "extract a value from a given index";
	let hasVerifier = 1;
}

def Attr_ConstantOp : Op<Arrays_Dialect, "constant", [Pure, ConstantLike]> {
	let	summary = "Define a constnat int";
	let arguments = (ins AnyAttr:$value);
	let results = (outs AnyType:$output);
	let hasCustomAssemblyFormat = 1;

	//let hasVerifier = 1;
	//let hasFolder = 1;
}

#endif
