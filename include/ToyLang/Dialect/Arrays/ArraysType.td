#ifndef LIB_DIALECT_ARR_POLYTYPES_TD_
#define LIB_DIALECT_ARR_POLYTYPES_TD_

include "ArraysDialect.td"
include "ArraysInterface.td"
include "../Primitive/PrimitiveInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class Arr_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Arrays_Dialect, name,traits>{
	let mnemonic = typeMnemonic;
}

def Arr_ArrType : Arr_Type<"Array","array", [DeclareTypeInterfaceMethods<ArraysTypeInterface>]> {
	let summary = "Array type";
	let description = [{
		Array type.
	}];

	let parameters = (ins "unsigned":$length, "PrimitiveTypeInterface":$type);
	let assemblyFormat = "`<` $length `,` $type `>`";

}
#endif
