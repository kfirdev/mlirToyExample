{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.llvmPackages_19.libllvm
    pkgs.llvmPackages_19.mlir
	pkgs.zlib
	pkgs.cmake
	pkgs.libffi
    pkgs.libxml2
    pkgs.cudaPackages.cudatoolkit
	pkgs.clang
  ];

  CUDAToolkit_ROOT="${pkgs.cudatoolkit}";
  shellHook = ''
	export CC=clang
	export CXX=clang++
  '';
}


