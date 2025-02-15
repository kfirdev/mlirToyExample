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
AllTypesMatch - meaning all the variables you mentioned have the same type.
```tablegen
AllTypesMatch<["lowerBound", "upperBound", "step"]>,
```

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


## Other traits
Here are some traits which don't have an explanation on the site and their explanation:

RecursiveMemoryEffects - This trait indicates that the memory effects of an operation includes the
effects of operations nested within its regions. If the operation has no derived effects interfaces, 
the operation itself can be assumed to have no memory effects.

RecursivelySpeculatable - This trait marks an op (which must be tagged as implementing the 
ConditionallySpeculatable interface) as being recursively speculatable.
This means that said op can be speculated only if all the instructions in all the 
regions attached to the op can be speculated. 

NoRegionArguments - This trait provides a verifier for ops that are expecting their regions to
not have any arguments 

HasParent - This class provides a verifier for ops that are expecting their parent to be one of the given parent ops. 
for example using HasParent<"FuncOp"> meaning this op has to exist in a region owned by the Func op
