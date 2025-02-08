#ifndef ARRAYS_INTERFACES_TD_
#define ARRAYS_INTERFACES_TD_

include "mlir/IR/OpBase.td"

def ArraysAttrInterface : AttrInterface<"ArraysAttrInterface">{
	let description = [{
		This is an interface it interact with all of the attributes in the array dialect.
	}];

	let methods = [
		InterfaceMethod<
		"Get the width of the value",
		"unsigned", "getWidth",(ins)>,
		InterfaceMethod<
		"Add two primitive interface attributes",
		"ArraysAttrInterface", "add",(ins "ArraysAttrInterface&":$other)>,
		InterfaceMethod<
		"Subtract two primitive interface attributes",
		"ArraysAttrInterface", "sub",(ins "ArraysAttrInterface&":$other)>,
		InterfaceMethod<
		"Multiply two primitive interface attributes",
		"ArraysAttrInterface", "mult",(ins "ArraysAttrInterface&":$other)>,
		InterfaceMethod<
		"Divide two primitive interface attributes",
		"ArraysAttrInterface", "div",(ins "ArraysAttrInterface&":$other)>,
		InterfaceMethod<
		"Convert to standard",
		"mlir::Operation*", "toStandard",(ins "ConversionPatternRewriter&":$rewriter, "mlir::Location":$loc)>,
	];


}

def ArraysTypeInterface : TypeInterface<"ArraysTypeInterface">{
	let description = [{
		This is an interface it interact with all of the types in the arrays dialect.
	}];
	let methods = [
		InterfaceMethod<
		"Get the width of the value",
		"unsigned", "getWidth",(ins)>,
		InterfaceMethod<
		"Get the length of the array",
		"unsigned int", "getLength",(ins)>,
		InterfaceMethod<
		"Get type with length of this and another type combined",
		"mlir::Type", "getCombined",(ins "unsigned":$other_length)>,
		//InterfaceMethod<
		//"Get inner type",
		//"mlir::Type", "getCombined",(ins "unsigned":$other_length)>
	];
}

#endif
