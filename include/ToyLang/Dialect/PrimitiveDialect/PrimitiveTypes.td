#ifndef LIB_DIALECT_PRIM_POLYTYPES_TD_
#define LIB_DIALECT_PRIM_POLYTYPES_TD_

include "PrimitiveDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Prim_Type<string name, string typeMnemonic> : TypeDef<Primitive_Dialect, name>{
	let mnemonic = typeMnemonic;
}

def Prim_IntegerType : Prim_Type<"Integer","int"> {
	let summary = "Integer type with arbitrary precision up to a fixed limit";
	let description = [{
		Integer types with a given precision.
	}];

	//let parameters = (ins "unsigned":$width);
}
#endif
