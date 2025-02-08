#ifndef LIB_DIALECT_ARR_ARRATTR_TD_
#define LIB_DIALECT_ARR_ARRATTR_TD_

include "ArraysDialect.td"
include "../Primitive/PrimitiveInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class Arrays_Attr<string name, string attrMnemonic, list<Trait> traits = []> : AttrDef<Arrays_Dialect, name, traits>{
	let mnemonic = attrMnemonic;
}

def Arr_ArrAttr : Arrays_Attr<"Array", "array"> {
	let summary = "An attribute containing an array";
	let description = [{
		Contains an Arrays
	}];

	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::SmallVector<PrimitiveAttrInterface>":$value);
	let hasCustomAssemblyFormat = 1;
}
#endif
