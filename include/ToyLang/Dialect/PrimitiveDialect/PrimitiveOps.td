#ifndef LIB_DIALECT_PRIM_PRIMOPS_TD_
#define LIB_DIALECT_PRIM_PRIMOPS_TD_

include "mlir/IR/OpBase.td"
include "PrimitiveDialect.td"
include "PrimitiveTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PrimInt_AddOp : Op<Primitive_Dialect, "add", [Pure,SameOperandsAndResultElementType]> {
	let summary = "Addition operation between to ints";
	let arguments = (ins Prim_IntegerType:$lhs, Prim_IntegerType:$rhs);
	let results = (outs Prim_IntegerType:$output);
	let assemblyFormat =  "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}

#endif
