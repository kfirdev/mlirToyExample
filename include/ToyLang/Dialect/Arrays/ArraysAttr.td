#ifndef LIB_DIALECT_ARR_ARRATTR_TD_
#define LIB_DIALECT_ARR_ARRATTR_TD_

include "ArraysDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Arrays_Attr<string name, string attrMnemonic, list<Trait> traits = []> : AttrDef<Arrays_Dialect, name, traits>{
	let mnemonic = attrMnemonic;
}

def Arr_IntegerArrAttr : Arrays_Attr<"IntegerArr", "intArr"> {
	let summary = "An attribute containing a Integer Array";
	let description = [{
		An integer Array attribute is a literal attribute that represents an integral
    	value of the specified integer array.
	}];

	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::SmallVector<primitive::IntegerAttr>":$value);
	let hasCustomAssemblyFormat = 1;
}

def Arr_FloatArrAttr : Arrays_Attr<"FloatArr", "floatArr"> {
	let summary = "An attribute containing a Float Array";
	let description = [{
		float array attribute
	}];

	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::SmallVector<primitive::FloatAttr>":$value);
	let hasCustomAssemblyFormat = 1;
}

def Arr_BoolArrAttr : Arrays_Attr<"BoolArr", "boolArr"> {
	let summary = "An attribute containing a Bool Array";
	let description = [{
		bool array attribute
	}];

	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::SmallVector<primitive::BoolAttr>":$value);
	let hasCustomAssemblyFormat = 1;
}
#endif
