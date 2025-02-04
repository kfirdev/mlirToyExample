#ifndef LIB_DIALECT_PRIM_PRIMATTR_TD_
#define LIB_DIALECT_PRIM_PRIMATTR_TD_

include "PrimitiveDialect.td"
include "PrimitiveInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
//include "mlir/IR/BuiltinAttributes.td"

class Primitive_Attr<string name, string attrMnemonic, list<Trait> traits = []> : AttrDef<Primitive_Dialect, name, traits>{
	let mnemonic = attrMnemonic;
}

def Prim_IntegerAttr : Primitive_Attr<"Integer", "int", [DeclareAttrInterfaceMethods<PrimitiveAttrInterface>]> {
	let summary = "An attribute containing a Integer value";
	let description = [{
		An integer attribute is a literal attribute that represents an integral
    	value of the specified integer type.
	}];

	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);
    let assemblyFormat = "$value";

	let builders = [
	  AttrBuilderWithInferredContext<(ins "Type":$type,"const APInt &":$value)>
  	];

    let genVerifyDecl = 1;

	let skipDefaultBuilders = 1;
}

def Prim_FloatAttr : Primitive_Attr<"Float", "float", [DeclareAttrInterfaceMethods<PrimitiveAttrInterface>]> {
	let summary = "An attribute containing a float value";
	let description = [{
		A float attribute is a literal attribute that represents an integral
    	value of the specified float type.
	}];

	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APFloat":$value);
    let assemblyFormat = "$value";

	let builders = [
	  AttrBuilderWithInferredContext<(ins "Type":$type,"const APFloat &":$value)>,
	  AttrBuilder<(ins "Type":$type,"const APFloat &":$value)>
  	];

    //let genVerifyDecl = 1;

	let skipDefaultBuilders = 1;
}
def Prim_BoolAttr : Primitive_Attr<"Bool", "bool", [DeclareAttrInterfaceMethods<PrimitiveAttrInterface>]> {
	let summary = "An attribute containing a bool value";
	let description = [{
		A bool attribute is a literal attribute that represents a bool value.
	}];

	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "bool":$value);
    let assemblyFormat = "$value";

	let builders = [
	  AttrBuilderWithInferredContext<(ins "Type":$type,"bool":$value)>,
	  AttrBuilder<(ins "Type":$type,"bool":$value)>
  	];

    //let genVerifyDecl = 1;

	let skipDefaultBuilders = 1;
}
def Prim_StringAttr : Primitive_Attr<"String", "string", [DeclareAttrInterfaceMethods<PrimitiveAttrInterface>]> {
	let summary = "An attribute containing a string value";
	let description = [{
		A string attribute is a literal attribute that represents a string value.
	}];

	//let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::ArrayRef<IntegerAttr>":$value);
	let parameters = (ins AttributeSelfTypeParameter<"">:$type, "llvm::SmallVector<IntegerAttr>":$value);
	//let parameters = (ins AttributeSelfTypeParameter<"">:$type, "std::string":$value);
    //let assemblyFormat = "$value";
	let hasCustomAssemblyFormat = 1;
	//let skipDefaultBuilders = 1;

	//let builders = [
	//  AttrBuilderWithInferredContext<(ins "Type":$type,"llvm::SmallVector<IntegerAttr>":$value)>,
	//  AttrBuilder<(ins "Type":$type,"llvm::SmallVector<IntegerAttr>":$value)>
  	//];
	//let skipDefaultBuilders = 1;

    //let genVerifyDecl = 1;

}
#endif
