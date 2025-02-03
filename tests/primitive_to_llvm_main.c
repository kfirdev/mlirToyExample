#include <stdio.h>
#include <stdbool.h>

int test_primitive_fn(int x);
float test_primitive_fn_double(float x);
bool test_bool(bool x,bool y);

int main(int argc, char *argv[]){
	int i = 1;
	int result = test_primitive_fn(i);
	printf("Result: %d\n", result);
	float f = 2.64;
	float resultdb = test_primitive_fn_double(f);
	printf("Result: %f\n", resultdb);
	bool x = false;
	bool y = true;
	bool resultbool = test_bool(x,y);
	printf("Result: %s\n", resultbool ? "true" : "false");
	return 0;
}
