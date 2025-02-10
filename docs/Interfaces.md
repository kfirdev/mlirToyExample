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
