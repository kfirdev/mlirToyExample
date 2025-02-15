#ifndef LIB_DIALECT_PRIM_PRIMOPS_TD_
#define LIB_DIALECT_PRIM_PRIMOPS_TD_

include "PrimitiveDialect.td"
include "PrimitiveTypes.td"
include "PrimitiveAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

class Primitive_BinOpBase<string name> : Op<Primitive_Dialect, name, [Pure,SameOperandsAndResultType]>{
	let assemblyFormat =  "$lhs `,` $rhs attr-dict `:` qualified(type($output))";
	let hasFolder = 1;
	let hasVerifier = 1;
}

class Primitve_IntOpBase<string name> : Primitive_BinOpBase<name>{
	let arguments = (ins PrimitiveTypeInterface:$lhs, PrimitiveTypeInterface:$rhs);
	let results = (outs PrimitiveTypeInterface:$output);
}


def PrimInt_AddOp : Primitve_IntOpBase<"add"> {
	let summary = "Addition operation between to ints";
}
def PrimInt_MultOp : Primitve_IntOpBase<"mul"> {
	let summary = "Addition operation between to ints";
}
def PrimInt_SubOp : Primitve_IntOpBase<"sub"> {
	let summary = "Addition operation between to ints";
}
def PrimInt_DivOp : Primitve_IntOpBase<"div"> {
	let summary = "Addition operation between to ints";
}

def Prim_FromStandardOp : Op<Primitive_Dialect, "from_standard", [Pure]> {
  let summary = "Creates a Primitive type from standard type";
  let arguments = (ins AnyTypeOf<[AnyInteger,AnyFloat]>:$input);
  let results = (outs PrimitiveTypeInterface:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($output))";
  //let hasFolder = 1;
}

def Prim_ToStandardOp : Op<Primitive_Dialect, "to_standard", [Pure]> {
  let summary = "Converts a Primitive type to standard";
  let arguments = (ins PrimitiveTypeInterface:$input);
  let results = (outs AnyTypeOf<[AnyInteger,AnyFloat]>:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";
}
// DeclareOpInterfaceMethods<RegionBranchOpInterface>
def Prim_IfOp : Op<Primitive_Dialect, "if", [RecursiveMemoryEffects, RecursivelySpeculatable, NoRegionArguments]> {
  let summary = "if statement";
  let arguments = (ins Prim_BoolType:$condition);
  let results = (outs Variadic<PrimitiveTypeInterface>:$result);
  let regions = (region SizedRegion<1>:$thenRegion,
                         MaxSizedRegion<1>:$elseRegion);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
}
//ParentOneOf<["IfOp"]>
def Prim_YieldOp : Op<Primitive_Dialect, "yield", [Pure, ReturnLike, Terminator]> {
  let summary = "yield statement";
  let arguments = (ins Variadic<PrimitiveTypeInterface>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

def Prim_ForOp : Op<Primitive_Dialect, "for", [AutomaticAllocationScope,
  AllTypesMatch<["lowerBound", "higherBound", "step"]>,
  DeclareOpInterfaceMethods<LoopLikeOpInterface,["getRegionIterArgs","getInitsMutable"]>,
  RecursiveMemoryEffects, RecursivelySpeculatable]> {

  let summary = "for loop";
  let arguments = (ins 
  Prim_IntegerType:$lowerBound, 
  Prim_IntegerType:$higherBound,
  Prim_IntegerType:$step,
  Variadic<PrimitiveTypeInterface>:$initArgs
  );


  let results = (outs Variadic<PrimitiveTypeInterface>:$result);
  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
	  Value getInductionVar() { return getRegion().front().getArguments()[0]; }
  }];

  let hasCustomAssemblyFormat = 1;
  //let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def FuncOp : Op<Primitive_Dialect,"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "primitive.func" operation represents a user defined function. These are
    callable SSA-region operations that contain primitive computations.

    Example:

    ```mlir
    primitive.func @main() {
      %0 = primitive.constant dense<5.500000e+00> : tensor<f64>
      %1 = primitive.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      primitive.print %1 : tensor<2x2xf64>
      primitive.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}
//HasParent<"FuncOp">,

def ReturnOp : Op<Primitive_Dialect,"return", [Pure, Terminator,HasParent<"FuncOp">]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      primitive.func @foo() -> tensor<2xf64> {
        ...
        primitive.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<PrimitiveTypeInterface>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  //let hasVerifier = 1;
}


def PrimInt_ConstantOp : Op<Primitive_Dialect, "constant", [Pure, ConstantLike]> {
	let	summary = "Define a constnat int";
	//let arguments = (ins Prim_IntegerAttr:$value);
	let arguments = (ins PrimitiveAttrInterface:$value);
	//let arguments = (ins AnyI32Attr:$value);
	let results = (outs PrimitiveTypeInterface:$output);
	//let assemblyFormat = "$value attr-dict type($output)";
	//let assemblyFormat = "$value attr-dict `:` type($output)";
	let hasCustomAssemblyFormat = 1;

	let builders = [
		OpBuilder<(ins "Type":$type, "PrimitiveAttrInterface":$value)>
	];
	let skipDefaultBuilders = 1;

	let hasVerifier = 1;
	let hasFolder = 1;
}

#endif
