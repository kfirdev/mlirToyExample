#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct {
    char    *field0;   // corresponds to the first 'ptr'
    char    *field1;   // corresponds to the second 'ptr'
    int64_t field2;   // corresponds to the i64
    int64_t field3[1]; // corresponds to [1 x i64]
    int64_t field4[1]; // corresponds to [1 x i64]
} test_string_ret_t;

// Declare the function as external.
extern test_string_ret_t test_string(void);

int test_primitive_fn(int x);
float test_primitive_fn_double(float x);
bool test_bool(bool x,bool y);
//char* test_string(char* inp1, char* inp2);

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
	test_string_ret_t resultstr = test_string();
	printf("Result: %s\n",resultstr.field0);
	return 0;
}
