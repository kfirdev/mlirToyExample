## Overview
Types in MLIR are representing types, the don't hold any the actual value
but rather just the outline.
On the other hand attributes are holding the actual value.

Traits and interfaces can be attached to types and attributes for adding additional functions and behaviors.

Custom parser could be added to attributes and types so they could be parsed in any way you want.

For further reading checkout [AttributesAndTypes](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)
