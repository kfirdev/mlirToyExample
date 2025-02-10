## Overview
Operations are exactly what they sound the are operations between different things in the program,
for example addition, subtraction and any operation you want it doesn't have to binary or anything,
it could be anything from that to what ever you want.

Interfaces and traits can be attached to operations which adds additional functions and verification respectively.

Custom verifies can be added to ops to verify types where inserted properly.
Custom folders can be added to ops in order to propagate constants and pre-compute what ever possible.
Custom parser could be added to ops so it could be parsed in any way you want.

Operations are mostly defined using table gen check out [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/) for further reading on ops and defining them with table gen.

