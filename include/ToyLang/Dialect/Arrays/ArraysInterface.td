#ifndef ARRAYS_INTERFACES_TD_
#define ARRAYS_INTERFACES_TD_

include "mlir/IR/OpBase.td"

def ArrayInferTypeInterface : OpInterface<"ArrayInferTypeInterface">{
	let description = [{
		To infer type
	}];
	let methods = [
		StaticInterfaceMethod<
		"infer type",
		"::llvm::LogicalResult", "inferReturnTypes",
		(ins "::mlir::MLIRContext*":$context, "::std::optional<::mlir::Location>":$location, "::mlir::ValueRange":$operands, "::mlir::DictionaryAttr":$attributes, "::mlir::OpaqueProperties":$properties, "::mlir::RegionRange":$regions, "::llvm::SmallVectorImpl<::mlir::Type>&":$inferredReturnTypes)>,
	];
}

#endif
