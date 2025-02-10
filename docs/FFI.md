## General
In general in order to use the ffi you convert the file to a library after that link it to your executable.
Working with the file is the ffi for the language.

## Basic types
For basic types you can use the equivalent in your chosen language for examples look at the c files in the test folder.

## Memref
For memref you would have to use a struct or a class because a memref encompasses data in it.
Here is an explanation from chatgpt.
### Background

When MLIR lowers a memref type to LLVM IR, it “destructures” the memref into several values. For a one-dimensional memref (e.g., memref<10xf32>), you get a function signature like this in LLVM IR:

```llvm
define void @example(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64)
```

These correspond to:  
- **%arg0 (!llvm.ptr):** Pointer to the allocated memory block.  
- **%arg1 (!llvm.ptr):** Aligned pointer (the pointer adjusted to meet alignment requirements).  
- **%arg2 (i64):** Offset into the allocated memory where the memref “view” begins.  
- **%arg3 (i64):** Size (number of elements) in the first (and only) dimension.  
- **%arg4 (i64):** Stride, which indicates how many elements to skip to move from one element to the next.

---

### Example in C

Imagine you have the following MLIR function:

```mlir
func @example(%arg0: memref<10xf32>) {
  // Use the memref...
  return
}
```

After lowering, a corresponding C prototype might be declared as:

```c
// The lowered function prototype in C
void example(void *allocated, void *aligned, long offset, long size, long stride);
```

Suppose in your C code you have an array of 10 floats:

```c
#include <stdio.h>

float data[10] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
```

You would then call the function using the descriptor fields extracted from the memref. For a typical contiguous array:
- **Allocated pointer (`allocated`):** The base pointer, which is the address of `data` (e.g., `data` or `&data[0]`).
- **Aligned pointer (`aligned`):** For a simple allocation where `data` is already suitably aligned, this is the same as the allocated pointer.
- **Offset (`offset`):** Since the memref starts at the beginning of the array, the offset is 0.
- **Size (`size`):** The number of elements in the memref. In this case, 10.
- **Stride (`stride`):** For a contiguous 1D array, the stride is 1 (meaning each successive element is one element apart in memory).

A sample call might then look like this:

```c
// Example call with illustrative values:
example((void*)data, (void*)data, 0, 10, 1);
```

If you were to imagine the values at runtime (using hypothetical pointer addresses), they might be:

- **allocated:** 0x7fffc000  
- **aligned:** 0x7fffc000  
- **offset:** 0  
- **size:** 10  
- **stride:** 1

These values ensure that the lowered function has all the information it needs to correctly access and index the memref data.
