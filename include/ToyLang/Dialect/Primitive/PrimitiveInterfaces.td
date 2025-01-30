#ifndef PRIMITIVE_INTERFACES_TD_
#define PRIMITIVE_INTERFACES_TD_

include "mlir/IR/OpBase.td"
// Just in general in order to use these interfaces you do a dyn_cast on a pointer to the interface,
// this doesn't require any extra allocations.

// You should create an interface that all of your attributes will have in order to interact with them in a general way.
// This means that this interface should have all of the function needed for the general interactions.
// Functions:
// 1. getValue to return the value as a string just for printing.
//def AttrInterface : AttrInterface<"AttrInterface">{
//}

// An interface should also be created for the types in order to the width and stuff needed by the types.
// 1. getWidth returns unsigned int which is the width.

#endif
