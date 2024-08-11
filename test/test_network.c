#include <unity.h>
#include "stdbool.h"

void setUp()
{

}

void tearDown()
{

}

void always_true(void){
    TEST_ASSERT_TRUE(true);
}

int main(void)
{
    UNITY_BEGIN();

    /*
     * Add in between
     * RUN_TEST(func_name); 
     */

    UNITY_END();

    return 0;
}