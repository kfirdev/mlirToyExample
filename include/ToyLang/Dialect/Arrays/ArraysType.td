#ifndef LIB_DIALECT_ARR_POLYTYPES_TD_
#define LIB_DIALECT_ARR_POLYTYPES_TD_

include "ArraysDialect.td"
include "ArraysInterface.td"
include "mlir/IR/AttrTypeBase.td"

class Arr_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Arrays_Dialect, name,traits>{
	let mnemonic = typeMnemonic;
}

def Arr_IntegerArrType : Arr_Type<"IntegerArr","intArr", [DeclareTypeInterfaceMethods<ArraysTypeInterface>]> {
	let summary = "Integer Array type with arbitrary precision up to a fixed limit";
	let description = [{
		Integer Array types with a given precision.
	}];

	let parameters = (ins "unsigned":$length, "unsigned":$width);
	let assemblyFormat = "`<` $length `,` $width `>`";

}
def Arr_FloatArrType : Arr_Type<"FloatArr","floatArr",[DeclareTypeInterfaceMethods<ArraysTypeInterface>]> {
	let summary = "Float Array type with arbitrary precision up to a fixed limit";
	let description = [{
		Float Array types with a given precision.
	}];

	let parameters = (ins "unsigned":$length, "unsigned":$width);
	let assemblyFormat = "`<` $length `,` $width `>`";

}
def Arr_BoolArrType : Arr_Type<"BoolArr","boolArr",[DeclareTypeInterfaceMethods<ArraysTypeInterface>]> {
	let summary = "Bool Array type with arbitrary precision up to a fixed limit";
	let description = [{
		Bool Array types with a given precision.
	}];

	let parameters = (ins "unsigned":$length);
	let assemblyFormat = "`<` $length `>`";

}
#endif
