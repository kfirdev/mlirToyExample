#include "ToyLang/Passes/Arrays/Passes.h"
#include "include/ToyLang/Dialect/Primitive/PrimitiveDialect.h"
#include "include/ToyLang/Dialect/Arrays/ArraysDialect.h"
#include "include/ToyLang/Conversions/General/AllToStandard.h"
#include "include/ToyLang/Passes/Primitive/Passes.h"
#include "include/ToyLang/Passes/Primitive/PrintPass.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
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

//void primitiveToLLVMPipelineBuilder(mlir::OpPassManager &manager){
//	manager.addPass(mlir::createCanonicalizerPass());
//	manager.addPass(mlir::toylang::primitive::createPrimToStandard());
//	manager.addPass(mlir::createCanonicalizerPass());
//	manager.addPass(mlir::createConvertSCFToCFPass());
//	manager.addPass(mlir::createConvertControlFlowToLLVMPass());
//	manager.addPass(mlir::createConvertFuncToLLVMPass());
//	manager.addPass(mlir::createArithToLLVMConversionPass());
//}

void ToyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::toylang::createAllToStandard());
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

  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createArithToLLVMConversionPass());
  manager.addPass(mlir::createConvertIndexToLLVMPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  manager.addPass(mlir::createConvertFuncToLLVMPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

  // Cleanup passes.
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}
void primitiveInliner(mlir::OpPassManager &manager){
	manager.addPass(mlir::createInlinerPass());
}

int main(int argc, char **argv) {
	mlir::DialectRegistry registry;
	registry.insert<mlir::toylang::primitive::PrimitiveDialect>();
	registry.insert<mlir::toylang::arrays::ArraysDialect>();
	mlir::registerAllDialects(registry);
	mlir::registerAllPasses();
	mlir::PassRegistration<mlir::toylang::primitive::FullUnrollPass>();
	mlir::PassRegistration<mlir::toylang::primitive::HoistConstPass>();
	mlir::PassRegistration<mlir::toylang::primitive::ShapeInfrencePass>();
	mlir::toylang::primitive::passes::registerPrintPass();
	mlir::toylang::registerAllToStandardPass();
	//mlir::toylang::arrays::registerArrToStandardPass();
	mlir::toylang::arrays::passes::registerConcatReplacePass();
	

	mlir::PassPipelineRegistration<>("toy-to-llvm",
			"Run passes to lower all dialects in toylang to llvm",
			ToyToLLVMPipelineBuilder);

	mlir::PassPipelineRegistration<>("prim-inline",
			"Run passes to inline all functions",
			primitiveInliner);

	return mlir::asMainReturnCode(
  	    mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}

