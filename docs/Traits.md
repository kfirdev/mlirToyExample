## Overview
Traits lets you specify things about your attribute, op or type,
some types are used as verifies verifying something is the way it
supposed to be for example SameOperandsAndResultType verifies that the 
operands and the result have the same type, other verifiers are just "slapped on"
meaning they don't actually check anything but by putting them MLIR knows it can perform
certain changes to optimize performance for example Pure trait that means that you don't change
any memory with this operation.

In addition you can create your own custom traits, custom traits are mostly for verification,
for example you can check out the [mlir beginners](https://www.jeremykun.com/2023/09/13/mlir-verifiers/#a-trait-based-custom-verifier)
where he shows a custom trait for verifying something, you could also have a trait that is just slapped on,
for example you can check out the IsABool and such traits here that are slapped on for me to change things
based on if it has the trait or not.

Moreover there are traits that can help you infer types:
AllTypesMatch - meaning you only need to specify one type.
TypesMatchWith - using one type to infer another, example:

Here I specify that the dest and result have the same type.
```tablegen
TypesMatchWith<"result type matches type of dest",
"dest", "result",
"$_self">
```

Here dest is the array type with contain the type of the elements and here I specify
that the type of the scalar is the same as the elements of dest.
```tablegen
TypesMatchWith<"scalar type matches type of an element in dest",
"dest", "scalar",
"mlir::cast<ArrayType>($_self).getType()">
```
SameTypeOperands - The operands have the same type.
SameOperandsAndResultType - the operands and result of the same type.

Another trait that help infer only the result type is InferTypeOpInterface
this trait added to an op that has interReturnType function uses that function
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
