get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

get_property(dialect_libs_llvm GLOBAL PROPERTY LLVM_LIBS)

message("message" ${dialect_libs_llvm})

set (LIBS
    ${dialect_libs}
    ${conversion_libs}
	MLIROptLib
	MLIRPrinPass
	MLIRPrim
	MLIRPrimLowerPass
	LLVMExecutionEngine
	MLIRExecutionEngine
	MLIRExecutionEngineUtils
	MLIRCAPIExecutionEngine
	#mlir_c_runner_utils
)

add_llvm_executable(toy-opt main.cpp)

llvm_update_compile_flags(toy-opt)
target_link_libraries(toy-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(toy-opt)

