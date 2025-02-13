#include <stdio.h>
#include <stdbool.h>

int test_primitive_fn(int x);
float test_primitive_fn_double(float x);
bool test_bool(bool x,bool y);
int test_if(bool cond,int res);
int test_for(int start, int end);

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

	bool cond = true;
	int res = 10;
	int resultIf = test_if(cond,res);
	printf("Result: %d\n", resultIf);

	int start = 1;
	int end = 10;
	int resultFor = test_for(start,end);
	printf("Result: %d\n", resultFor);
	return 0;
}
