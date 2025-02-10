## Overview
Bufferization in MLIR is the process of converting ops with tensor semantics to ops with memref semantics.
Bufferization is the last step before lowering to llvm.

## Memref overview
A memref is built from five things:
Allocated pointer (allocated): The base pointer, which is the address of data (e.g., data or &data[0]).
Aligned pointer (aligned): For a simple allocation where data is already suitably aligned, this is the same as the allocated pointer.
Offset (offset): Since the memref starts at the beginning of the array, the offset is 0.
Size (size): The number of elements in the memref. In this case, 10.
Stride (stride): For a contiguous 1D array, the stride is 1 (meaning each successive element is one element apart in memory).

For more info checkout FFI.md.


## Explanation 
In general bufferization converts tensor to buffer or in the case of mlir memref which are buffer like.
By default MLIR will lower buffer to fully dynamic ones which means that when using the copy operation it will
use a custom copy function @memrefCopy which in order to use you have to link a library to your executable.
In addition dynamic memrefs are probably less efficient because as we know the more static something the faster and easier 
it is to optimize it is.

In order to avoid that you will have to set it so it will convert function boundaries types to identity
maps which is the most static type.

```cpp
  bufferizationOptions.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
```

If the bufferization wasn't able to convert it to a memref it will convert it to fully dynamic
which is why it is important to also specify that when encountering an unknown type it will be 
converted to a fully static type.

```cpp
  bufferizationOptions.unknownTypeConverterFn = [=](
		  mlir::Value value, mlir::Attribute memorySpace,
		  const mlir::bufferization::BufferizationOptions &options) {
		auto tensorType = llvm::cast<mlir::TensorType>(value.getType());
      return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                  memorySpace);
  };
```

## Function boundaries types
By default, function boundaries are not bufferized. This is because there are currently limitations around function graph bufferization: recursive calls are not supported. As long as there are no recursive calls, function boundary bufferization can be enabled with bufferize-function-boundaries. Each tensor function argument and tensor function result is then turned into a memref. The layout map of the memref type can be controlled with function-boundary-type-conversion.

