#include "ToyLang/Passes/Arrays/Passes.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysDialect.h"
#include "include/ToyLang/Conversions/Primitive/PrimitiveToStandard.h"
#include "include/ToyLang/Passes/Primitive/Passes.h"
#include "include/ToyLang/Passes/Primitive/PrintPass.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitCPass.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "include/ToyLang/Conversions/Arrays/ArraysToStandard.h"

void arraysToStandardPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::toylang::arrays::createArrToStandard());
  manager.addPass(mlir::toylang::primitive::createPrimToStandard());
}

void arraysToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::toylang::arrays::createArrToStandard());
  manager.addPass(mlir::toylang::primitive::createPrimToStandard());
  manager.addPass(mlir::createCanonicalizerPass());

  manager.addPass(mlir::tosa::createTosaToTensor());
  manager.addPass(mlir::affine::createSimplifyAffineStructuresPass());
  manager.addPass(mlir::createLowerAffinePass());

  manager.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;

  bufferizationOptions.unknownTypeConverterFn = [=](
		  mlir::Value value, mlir::Attribute memorySpace,
		  const mlir::bufferization::BufferizationOptions &options) {
		auto tensorType = llvm::cast<mlir::TensorType>(value.getType());
      return mlir::bufferization::getMemRefTypeWithStaticIdentityLayout(tensorType,
                                                                  memorySpace);
  };

  // *SUPER IMPORTANT* this makes the memref not be so dynamic which then makes
  // the lowering not use the memrefCopy function this is very good
  bufferizationOptions.setFunctionBoundaryTypeConversion(mlir::bufferization::LayoutMapOption::IdentityLayoutMap);

  manager.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);
  manager.addPass(mlir::bufferization::createPromoteBuffersToStackPass());

  manager.addPass(mlir::memref::createExpandStridedMetadataPass());
  manager.addPass(mlir::createBufferizationToMemRefPass());

  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertIndexToLLVMPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

  // Cleanup passes.
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}


void primitiveToLLVMPipelineBuilder(mlir::OpPassManager &manager){
	manager.addPass(mlir::toylang::primitive::createPrimToStandard());
	manager.addPass(mlir::createCanonicalizerPass());
	manager.addPass(mlir::createConvertFuncToLLVMPass());
	manager.addPass(mlir::createArithToLLVMConversionPass());
}

int main(int argc, char **argv) {
	mlir::DialectRegistry registry;
	registry.insert<mlir::toylang::primitive::PrimitiveDialect>();
	registry.insert<mlir::toylang::arrays::ArraysDialect>();
	mlir::registerAllDialects(registry);
	mlir::registerAllPasses();
	mlir::PassRegistration<mlir::toylang::primitive::FullUnrollPass>();
	mlir::toylang::primitive::passes::registerPrintPass();
	mlir::toylang::primitive::registerPrimToStandardPass();
	//mlir::toylang::arrays::registerArrToStandardPass();
	mlir::toylang::arrays::passes::registerConcatReplacePass();
	
	mlir::PassPipelineRegistration<>("primitive-to-llvm",
			"Run passes to lower primitive dialect to llvm",
			primitiveToLLVMPipelineBuilder);
	mlir::PassPipelineRegistration<>("arrays-to-llvm",
			"Run passes to lower primitive dialect to llvm",
			arraysToLLVMPipelineBuilder);
	mlir::PassPipelineRegistration<>("arrays-to-standard",
			"Run passes to lower primitive dialect to llvm",
			arraysToStandardPipelineBuilder);

	return mlir::asMainReturnCode(
  	    mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}

