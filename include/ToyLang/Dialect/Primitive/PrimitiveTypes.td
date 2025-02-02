#ifndef LIB_DIALECT_PRIM_POLYTYPES_TD_
#define LIB_DIALECT_PRIM_POLYTYPES_TD_

include "PrimitiveDialect.td"
include "PrimitiveInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class Prim_Type<string name, string typeMnemonic, list<Trait> traits = []> : TypeDef<Primitive_Dialect, name,traits>{
	let mnemonic = typeMnemonic;
}

def Prim_IntegerType : Prim_Type<"Integer","int",[DeclareTypeInterfaceMethods<PrimitiveTypeInterface>]> {
	let summary = "Integer type with arbitrary precision up to a fixed limit";
	let description = [{
		Integer types with a given precision.
	}];

	//let builders = [
	//    TypeBuilder<(ins "Type":$elementType, "unsigned":$width)>
	//];
	//let skipDefaultBuilders = 1;

	let parameters = (ins "unsigned":$width);
	let assemblyFormat = "`<` $width `>`";

}
def Prim_FloatType : Prim_Type<"Float","float",[DeclareTypeInterfaceMethods<PrimitiveTypeInterface>]> {
	let summary = "Float type with arbitrary precision up to a fixed limit";
	let description = [{
		Float types with a given precision.
	}];

	let parameters = (ins "unsigned":$width);
	let assemblyFormat = "`<` $width `>`";

}
#endif
