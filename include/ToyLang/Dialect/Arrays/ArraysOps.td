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

def Arr_ConcatOp : Op<Arrays_Dialect,"concat",[Pure,DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
	let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)`";
	let arguments = (ins Arr_ArrType:$lhs, Arr_ArrType:$rhs);
	let results = (outs Arr_ArrType:$output);
	let summary = "concat operation between to arrays";
	let hasVerifier = 1;
	let hasFolder = 1;
}

def Arr_ExtractOp : Op<Arrays_Dialect,"extract",[Pure,
				   TypesMatchWith<"result type matches type of dest",
                   "tensor", "output",
                   "mlir::cast<ArrayType>($_self).getType()">
				   ]> {
	let assemblyFormat = "$tensor `[` $indices `]` attr-dict `:` type($tensor)";
	let arguments = (ins Index:$indices, Arr_ArrType:$tensor);
	let results = (outs PrimitiveTypeInterface:$output);
	let summary = "extract a value from a given index";
	let hasVerifier = 1;
	let hasFolder = 1;
}
def Arr_InsertOp : Op<Arrays_Dialect,"insert",[
				   TypesMatchWith<"result type matches type of dest",
                   "dest", "result",
                   "$_self">,
				   TypesMatchWith<"result type matches type of dest",
                   "dest", "scalar",
                   "mlir::cast<ArrayType>($_self).getType()">
				   ]> {
	let assemblyFormat = "$scalar `into` $dest `[` $indices `]` attr-dict `:` type($dest)";
	let arguments = (ins Builtin_Index:$indices, PrimitiveTypeInterface:$scalar,Arr_ArrType:$dest);
	let results = (outs Arr_ArrType:$result);
	let summary = "extract a value from a given index";
	//let hasVerifier = 1;
	let hasFolder = 1;
}

def Attr_ConstantOp : Op<Arrays_Dialect, "constant", [Pure, ConstantLike]> {
	let	summary = "Define a constnat int";
	let arguments = (ins Arr_ArrAttr:$value);
	let results = (outs Arr_ArrType:$output);
	let hasCustomAssemblyFormat = 1;

	//let hasVerifier = 1;
	let hasFolder = 1;
}

#endif
