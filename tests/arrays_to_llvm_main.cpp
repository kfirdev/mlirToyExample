#include <array>
#include <iostream>
#include <ostream>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

template<typename tp, int length>
class llvm_struct{
	tp* alloc_ptr;
	tp* align_ptr;
	int64_t offset;
	int64_t size;
	int64_t stride;
	
	public:
		llvm_struct(std::array<tp,length>& arr){
			alloc_ptr = arr.data();
			align_ptr = arr.data();
			offset = 0;
			size = length;
			stride = 1;
		}
		llvm_struct(tp* ptr){
			alloc_ptr = ptr;
			align_ptr = ptr;
			offset = 0;
			size = length;
			stride = 1;
		}
		friend std::ostream& operator<<(std::ostream& os, const llvm_struct<tp,length>& llvm_strc){
			os << "["; 
			for (int i = llvm_strc.offset; i<llvm_strc.size;i+=llvm_strc.stride){
				os << llvm_strc.align_ptr[i];
				if (i != llvm_strc.size-1){
					os << ",";
				}
			}
			os << "]\n";
			return os;
		}
		friend llvm_struct<int,6> test_concatOp(llvm_struct<int,3> &first,llvm_struct<int,3> &second);
		friend int test_extractOp(llvm_struct<int,3> &first);
		friend llvm_struct<int,3> test_insertOp(llvm_struct<int,3> &first);

};

extern "C" {
llvm_struct<int,6> test_concatOp_extern(int*,int*,int64_t,int64_t,int64_t,int*,int*,int64_t,int64_t,int64_t);
int test_extractOp_extern(int*,int*,int64_t,int64_t,int64_t);
llvm_struct<int,3> test_insertOp_extern(int*,int*,int64_t,int64_t,int64_t);
}

llvm_struct<int,6> test_concatOp(llvm_struct<int,3> &first,llvm_struct<int,3> &second){
	return test_concatOp_extern(
			first.alloc_ptr,first.align_ptr,first.offset,first.size,first.stride,
			second.alloc_ptr,second.align_ptr,second.offset,second.size,second.stride);
}
int test_extractOp(llvm_struct<int,3> &first){
	return test_extractOp_extern(
			first.alloc_ptr,first.align_ptr,first.offset,first.size,first.stride);
}
llvm_struct<int,3> test_insertOp(llvm_struct<int,3> &first){
	return test_insertOp_extern(
			first.alloc_ptr,first.align_ptr,first.offset,first.size,first.stride);
}

int main(int argc, char *argv[]){
	int values[3] = {1,2,3};
	int values_other[3] = {4,5,6};
	llvm_struct<int,3> first {values};
	llvm_struct<int,3> second {values_other};

	llvm_struct<int,6> resConcat = test_concatOp(first,second);
	std::cout << "Result: " << resConcat;

	int resExtract = test_extractOp(first);
	printf("Result: %d\n",resExtract);

	llvm_struct<int,3> resInsert = test_insertOp(first);
	std::cout << "Result: " << first;
	return 0;
}
