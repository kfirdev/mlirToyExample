#ifndef LIB_DIALECT_PRIM_PRIMOPS_TD_
#define LIB_DIALECT_PRIM_PRIMOPS_TD_

include "PrimitiveDialect.td"
include "PrimitiveTypes.td"
include "PrimitiveAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

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

	  /// Return the `index`-th region iteration argument.
	  //BlockArgument getRegionIterArgs() {
	  //    return getRegion().front().getArguments().drop_front(1);
	  //}
  }];

  let hasCustomAssemblyFormat = 1;
  //let hasFolder = 1;
  let hasVerifier = 1;
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
