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

	let builders = [
  	  //AttrBuilderWithInferredContext<(ins "Type":$type,
  	  //                                    "const APInt &":$value), [{
	  //  auto integerType = mlir::dyn_cast<IntegerType>(type);
  	  //  llvm::APInt adjustedValue = value;
  	  //  if (value.getBitWidth() != integerType.getWidth()) {
  	  //    adjustedValue = value.zextOrTrunc(integerType.getWidth());
  	  //  }

  	  //  // Use the generated function to create the attribute.
  	  //  //return Base::get(type.getContext(), type, adjustedValue);

  	  //  return $_get(type.getContext(), type, adjustedValue);
  	  //}]>
	  AttrBuilderWithInferredContext<(ins "Type":$type,"const APInt &":$value)>
  	];

    let genVerifyDecl = 1;

	let skipDefaultBuilders = 1;
}
#endif
