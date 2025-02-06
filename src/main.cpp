#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Conversions/Primitive/PrimitiveToStandard.h"
#include "include/ToyLang/Passes/Primitive/Passes.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitCPass.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitCPass.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitCPass.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

void primitiveToLLVMPipelineBuilder(mlir::OpPassManager &manager){
	manager.addPass(mlir::toylang::primitive::createPrimToStandard());
	manager.addPass(mlir::createCanonicalizerPass());
	manager.addPass(mlir::toylang::primitive::createConcatReplacePass());

	// One-shot bufferize, from
	// https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  	mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  	bufferizationOptions.bufferizeFunctionBoundaries = true;
  	manager.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  	mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  	mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);
	//manager.addPass(mlir::bufferization::createPromoteBuffersToStackPass());

	manager.addPass(mlir::memref::createExpandStridedMetadataPass());

	manager.addPass(mlir::createBufferizationToMemRefPass());

	manager.addPass(mlir::createConvertFuncToLLVMPass());
	manager.addPass(mlir::createLowerAffinePass());
	manager.addPass(mlir::createArithToLLVMConversionPass());
	manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
	manager.addPass(mlir::createReconcileUnrealizedCastsPass());
	
	// Cleanup
	manager.addPass(mlir::createCanonicalizerPass());
	manager.addPass(mlir::createSCCPPass());
	manager.addPass(mlir::createCSEPass());
	manager.addPass(mlir::createSymbolDCEPass());
}

void primitiveToEmitCPipelineBuilder(mlir::OpPassManager &manager){
	manager.addPass(mlir::toylang::primitive::createPrimToStandard());
	manager.addPass(mlir::createCanonicalizerPass());
	manager.addPass(mlir::toylang::primitive::createConcatReplacePass());

	// One-shot bufferize, from
	// https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  	mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  	bufferizationOptions.bufferizeFunctionBoundaries = true;
  	manager.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  	mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  	mlir::bufferization::buildBufferDeallocationPipeline(manager, deallocationOptions);
	manager.addPass(mlir::bufferization::createPromoteBuffersToStackPass());

	manager.addPass(mlir::memref::createExpandStridedMetadataPass());

	manager.addPass(mlir::createBufferizationToMemRefPass());

	manager.addPass(mlir::createLowerAffinePass());
	manager.addPass(mlir::createConvertArithToEmitC());
	//manager.addPass(mlir::createConvertFuncToEmitC());
	//manager.addPass(mlir::createConvertMemRefToEmitC());
	//manager.addPass(mlir::createReconcileUnrealizedCastsPass());
	
	// Cleanup
	//manager.addPass(mlir::createCanonicalizerPass());
	//manager.addPass(mlir::createSCCPPass());
	//manager.addPass(mlir::createCSEPass());
	//manager.addPass(mlir::createSymbolDCEPass());
}
int main(int argc, char **argv) {
	mlir::DialectRegistry registry;
	registry.insert<mlir::toylang::primitive::PrimitiveDialect>();
	mlir::registerAllDialects(registry);
	mlir::registerAllPasses();
	//mlir::toylang::primitive::passes::registerPrintPass();
	mlir::toylang::primitive::passes::registerPasses();
	mlir::toylang::primitive::registerPrimToStandardPass();
	
	mlir::PassPipelineRegistration<>("primitive-to-llvm",
			"Run passes to lower primitive dialect to llvm",
			primitiveToLLVMPipelineBuilder);
	mlir::PassPipelineRegistration<>("primitive-to-emitc",
			"Run passes to lower primitive dialect to c",
			primitiveToEmitCPipelineBuilder);

	return mlir::asMainReturnCode(
  	    mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}

