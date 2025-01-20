get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# combining all of the libs
# only the last one is my library
set (LIBS
    ${dialect_libs}
    ${conversion_libs}
	MLIROptLib
	MLIRToy	
)

add_llvm_executable(toy-opt main.cpp)

llvm_update_compile_flags(toy-opt)
target_link_libraries(toy-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(toy-opt)

