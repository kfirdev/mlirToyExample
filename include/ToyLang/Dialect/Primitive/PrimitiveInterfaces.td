#ifndef PRIMITIVE_INTERFACES_TD_
#define PRIMITIVE_INTERFACES_TD_

include "mlir/IR/OpBase.td"
def PrimitiveAttrInterface : AttrInterface<"PrimitiveAttrInterface">{
	let description = [{
		This is an interface it interact with all of the attributes in the primitive dialect.
	}];

	let methods = [
		InterfaceMethod<
		"Get the value of an attribute as a String",
		"std::string", "getValueStr",(ins)>,
		InterfaceMethod<
		"Get the width of the value",
		"unsigned", "getWidth",(ins)>,
		InterfaceMethod<
		"Get the active bits/minimal bit width for this data",
		"unsigned", "getActiveWidth",(ins)>,
		InterfaceMethod<
		"Add two primitive interface attributes",
		"PrimitiveAttrInterface", "add",(ins "PrimitiveAttrInterface&":$other)>,
		InterfaceMethod<
		"Subtract two primitive interface attributes",
		"PrimitiveAttrInterface", "sub",(ins "PrimitiveAttrInterface&":$other)>,
		InterfaceMethod<
		"Multiply two primitive interface attributes",
		"PrimitiveAttrInterface", "mult",(ins "PrimitiveAttrInterface&":$other)>,
		InterfaceMethod<
		"Divide two primitive interface attributes",
		"PrimitiveAttrInterface", "div",(ins "PrimitiveAttrInterface&":$other)>,
		InterfaceMethod<
		"Convert to standard",
		"mlir::TypedAttr", "toStandard",(ins)>,
	];


}

def PrimitiveTypeInterface : TypeInterface<"PrimitiveTypeInterface">{
	let description = [{
		This is an interface it interact with all of the types in the primitive dialect.
	}];
	let methods = [
		InterfaceMethod<
		"Get the width of the value",
		"unsigned", "getWidth",(ins)>,
		InterfaceMethod<
		"Convert the type to a standard type",
		"mlir::Type", "toStandard",(ins)>,
		StaticInterfaceMethod<
		"Convert add to standard",
		"mlir::Operation*", "addToStandard",
		(ins "ConversionPatternRewriter&":$rewriter, "mlir::Location":$loc, "mlir::Value":$lhs, "mlir::Value":$rhs)>,
		StaticInterfaceMethod<
		"Convert sub to standard",
		"mlir::Operation*", "subToStandard",
		(ins "ConversionPatternRewriter&":$rewriter, "mlir::Location":$loc, "mlir::Value":$lhs, "mlir::Value":$rhs)>,
		StaticInterfaceMethod<
		"Convert mult to standard",
		"mlir::Operation*", "multToStandard",
		(ins "ConversionPatternRewriter&":$rewriter, "mlir::Location":$loc, "mlir::Value":$lhs, "mlir::Value":$rhs)>,
		StaticInterfaceMethod<
		"Convert div to standard",
		"mlir::Operation*", "divToStandard",
		(ins "ConversionPatternRewriter&":$rewriter, "mlir::Location":$loc, "mlir::Value":$lhs, "mlir::Value":$rhs)>,
	];
}

#endif
