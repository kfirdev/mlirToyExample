#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct {
	int* alloc_ptr;
	int* align_ptr;
	int64_t offset;
	int64_t size;
	int64_t stride;
} llvm_struct;


llvm_struct test_concatOp(int*,int*,int64_t,int64_t,int64_t,int*,int*,int64_t,int64_t,int64_t);
int test_extractOp(int*,int*,int64_t,int64_t,int64_t);
llvm_struct test_insertOp(int*,int*,int64_t,int64_t,int64_t);

int main(int argc, char *argv[]){
	int values[3] = {1,2,3};
	int values_other[3] = {4,5,6};

	llvm_struct resConcat = test_concatOp(values,values,0,3,1,values_other,values_other,0,3,1);
	printf("Result: ");
	printf("[");
	for (int i = resConcat.offset; i<resConcat.size;i+=resConcat.stride){
		printf("%d",resConcat.align_ptr[i]);
		if (i != resConcat.size-1){
			printf(",");
		}
	}
	printf("]\n");

	int resExtract = test_extractOp(values,values,0,3,1);
	printf("Result: %d\n",resExtract);

	llvm_struct resInsert = test_insertOp(values,values,0,3,1);
	printf("Result: ");
	printf("[");
	for (int i = 0; i<3;i++){
		printf("%d",values[i]);
		if (i != 2){
			printf(",");
		}
	}
	printf("]");
	return 0;
}
