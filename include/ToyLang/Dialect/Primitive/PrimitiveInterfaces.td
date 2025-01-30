#ifndef PRIMITIVE_INTERFACES_TD_
#define PRIMITIVE_INTERFACES_TD_

include "mlir/IR/OpBase.td"
// Just in general in order to use these interfaces you do a dyn_cast on a pointer to the interface,
// this doesn't require any extra allocations.

// You should create an interface that all of your attributes will have in order to interact with them in a general way.
// This means that this interface should have all of the function needed for the general interactions.
// Functions:
// 1. getValueStr to return the value as a string just for printing.
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
	];

}

// An interface should also be created for the types in order to the width and stuff needed by the types.
//def MyTypeInterface : TypeInterface<"MyTypeInterface">{
//}
// 1. getWidth returns unsigned int which is the width.

#endif
