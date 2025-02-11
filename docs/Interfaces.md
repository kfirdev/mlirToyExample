## Overview
Interfaces are basically a virtual base class you add virtual function by their definition
and each one you added can be used on the interface, you can have static and normal functions.

There are three types of interfaces: op interface, type interface and attribute interface,
each one is for the respective category.

Interfaces are very similar to traits in rust, but here every type, attribute or op that implements 
the interface can be casted to one and by that you can group them.

Interfaces could also force a type, attribute or op to implement an additional function using Declare(Attr|Op|Type)InterfaceMethods
respectively.
For further reading checkout [Interfaces](https://mlir.llvm.org/docs/Interfaces/)
And this [talk](https://www.youtube.com/watch?v=VCJAmOFvnh4)

## Some interfaces
Here are some explanation for interfaces which are not found on the docs:

RegionBranchOpInterface - This interface provides information for region operations that exhibit
branching behavior between held regions. I.e., this interface allows for
expressing control flow information for region holding operations.

This interface is meant to model well-defined cases of control-flow and
value propagation, where what occurs along control-flow edges is assumed to
be side-effect free.

A "region branch point" indicates a point from which a branch originates. It
can indicate either a region of this op or `RegionBranchPoint::parent()`. In
the latter case, the branch originates from outside of the op, i.e., when
first executing this op.

A "region successor" indicates the target of a branch. It can indicate
either a region of this op or this op. In the former case, the region
successor is a region pointer and a range of block arguments to which the
"successor operands" are forwarded to. In the latter case, the control flow
leaves this op and the region successor is a range of results of this op to
which the successor operands are forwarded to.

By default, successor operands and successor block arguments/successor
results must have the same type. `areTypesCompatible` can be implemented to
allow non-equal types.

Example:

```
%r = scf.for %iv = %lb to %ub step %step iter_args(%a = %b)
    -> tensor<5xf32> {
  ...
  scf.yield %c : tensor<5xf32>
}
```

`scf.for` has one region. The region has two region successors: the region
itself and the `scf.for` op. %b is an entry successor operand. %c is a
successor operand. %a is a successor block argument. %r is a successor
result.
 
InferTypeOpInterface - this interface added to an op that has interReturnType function uses that function
to infer the return type by that letting omit the return type from the assembly,
if the op doesn't implement interReturnType you can use DeclareOpInterfaceMethods<InferTypeOpInterface>
which will add this function to the op which you will have to implement yourself.
Here is an example implementation:

In this example the operands are arrays, the type of the output is the same as the input but
with a length that is equal to both them combined because this operation is a concat.
```cpp
::llvm::LogicalResult ConcatOp::inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  if (operands.size() <= 1)
    return ::mlir::failure();

  auto Lhs = mlir::dyn_cast<ArrayType>(operands[0].getType());
  auto Rhs = mlir::dyn_cast<ArrayType>(operands[1].getType());
  ::mlir::Type odsInferredType0 = ArrayType::get(operands[0].getType().getContext(),Lhs.getLength()+Rhs.getLength(),Lhs.getType());

  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}
```

InferTypeOpAdaptor - Convenient trait to define a wrapper to inferReturnTypes that passes in the Op Adaptor directly.

