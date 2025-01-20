{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
	pkgs.zlib
	pkgs.cmake
	pkgs.libffi
    pkgs.libxml2
    pkgs.cudaPackages.cudatoolkit
	#pkgs.clang
  ];

  CUDAToolkit_ROOT="${pkgs.cudatoolkit}";
  shellHook = ''
	export CC=clang
	export CXX=clang++
	export LLVM_BUILD_DIR=$HOME/personal/cpp_projects/llvm_project/llvm-project/build
  '';

}
