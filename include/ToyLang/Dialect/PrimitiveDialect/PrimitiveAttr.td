#ifndef LIB_DIALECT_PRIM_PRIMATTR_TD_
#define LIB_DIALECT_PRIM_PRIMATTR_TD_

include "mlir/IR/AttrTypeBase.td"
include "PrimitiveDialect.td"

class Primitive_Attr<string name, string attrMnemonic> : AttrDef<Primitive_Dialect, name>{
	let mnemonic = attrMnemonic;
}

def Prim_IntegerAttr : Primitive_Attr<"Integer", "int"> {
	let summary = "An attribute containing a Integer value";
	let description = [{
		An integer attribute is a literal attribute that represents an integral
    	value of the specified integer type.
	}];

	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);
    let assemblyFormat = "$value";
}



#endif
