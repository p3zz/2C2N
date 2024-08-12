#include <unity.h>
#include "stdbool.h"
#include "network.h"

void setUp()
{

}

void tearDown()
{

}

void test_always_true(void){
    TEST_ASSERT_TRUE(true);
}

void test_always_false(void){
    TEST_ASSERT_TRUE(false);
}

int main(void)
{
    UNITY_BEGIN();

    RUN_TEST(test_always_true);
    RUN_TEST(test_always_false);

    int result = UNITY_END();

    return result;
}