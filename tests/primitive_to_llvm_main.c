#include <stdio.h>

int test_primitive_fn(int x);
float test_primitive_fn_double(float x);

int main(int argc, char *argv[]){
	int i = 1;
	int result = test_primitive_fn(i);
	printf("Result: %d\n", result);
	float f = 2.64;
	float resultdb = test_primitive_fn_double(f);
	printf("Result: %f\n", resultdb);
	return 0;
}
