import lit.formats

config.name = "MLIRTests"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.excludes = ["CMakeLists.txt", "README.txt"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.test_source_root
config.environment["LIT_OPTS"] = "-v"


