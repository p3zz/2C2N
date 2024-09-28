#include <unity.h>
#include "matrix.h"

void setUp()
{

}

void tearDown()
{

}

void test_matrix2d_get_elem_as_ref(void){
    matrix2d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    const float* ref = matrix2d_get_elem_as_ref(&m1, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(13.5f, *ref);
}

void test_matrix2d_get_elem_as_mut_ref(void){
    matrix2d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    float* ref = matrix2d_get_elem_as_mut_ref(&m1, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(13.5f, *ref);
}

void test_matrix2d_get_elem(void){
    matrix2d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    float ref = matrix2d_get_elem(&m1, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(13.5f, ref);
}

void test_matrix2d_set_elem(void){
    matrix2d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,
        9.f, 13.5f, 1.5f
    };
    matrix2d_load(&m1, 2, 3, content);
    matrix2d_set_elem(&m1, 0, 2, 3.5f);
    float res = matrix2d_get_elem(&m1, 0, 2);
    TEST_ASSERT_EQUAL_FLOAT(3.5f, res);
}

void test_matrix3d_get_elem_as_ref(void){
    matrix3d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    const float* ref = matrix3d_get_elem_as_ref(&m1, 0, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(4.f, *ref);
}

void test_matrix3d_get_elem_as_mut_ref(void){
    matrix3d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    float* ref = matrix3d_get_elem_as_mut_ref(&m1, 0, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(4.f, *ref);
}

void test_matrix3d_get_elem(void){
    matrix3d_t m1 = {0};
    float content[] = {
        3.f, 1.f, 2.f,

        9.f, 4.f, 5.f
    };
    matrix3d_load(&m1, 1, 3, 2, content);
    float ref = matrix3d_get_elem(&m1, 0, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(4.f, ref);
}

void test_matrix3d_get_slice_as_mut_ref(void){
    matrix3d_t m1 = {0};
    matrix2d_t slice = {0};
    
    float content[] = {
        3.f, 1.f, 2.f,
        5.f, 6.f, 7.f,

        9.f, 4.f, 5.f,
        12.f, 11.f, 10.f
    };
    matrix3d_load(&m1, 2, 3, 2, content);
    matrix3d_get_slice_as_mut_ref(&m1, &slice, 1);
    float ref = matrix2d_get_elem(&slice, 1, 1);
    TEST_ASSERT_EQUAL_FLOAT(11.f, ref);
}

void test_common_matrix3d_reshape(void){
    matrix3d_t m = {0};
    matrix3d_t result = {0};

    float m_values[] = {
        4.f, 3.f,
        9.f, 1.f,
        5.f, 2.f,

        9.f, 2.f,
        7.f, 1.f,
        3.f, 2.f,
    };
    matrix3d_load(&m, 3, 2, 2, &m_values[0]);

    matrix3d_init(&result, 1, 12, 1);
    matrix3d_reshape(&m, &result);

    TEST_ASSERT_EQUAL_FLOAT(4.f, matrix3d_get_elem(&result, 0, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(1.f, matrix3d_get_elem(&result, 0, 3, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix3d_get_elem(&result, 0, 7, 0));
    TEST_ASSERT_EQUAL_FLOAT(7.f, matrix3d_get_elem(&result, 0, 8, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix3d_get_elem(&result, 0, 11, 0));

    matrix3d_destroy(&result);
}

void test_common_matrix3d_reshape_2(void){
    matrix3d_t m = {0};
    matrix3d_t result = {0};

    float m_values[] = {
        4.f, 3.f, 9.f, 1.f, 5.f, 2.f, 9.f, 2.f, 7.f, 1.f, 3.f, 2.f
    };

    matrix3d_load(&m, 1, 12, 1, &m_values[0]);
    matrix3d_init(&result, 3, 2, 2);

    matrix3d_reshape(&m, &result);
    TEST_ASSERT_EQUAL_FLOAT(4.f, matrix3d_get_elem(&result, 0, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(3.f, matrix3d_get_elem(&result, 0, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(9.f, matrix3d_get_elem(&result, 1, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix3d_get_elem(&result, 2, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(2.f, matrix3d_get_elem(&result, 0, 1, 1));
    TEST_ASSERT_EQUAL_FLOAT(3.f, matrix3d_get_elem(&result, 2, 0, 1));

    matrix3d_destroy(&result);
}

void test_matrix2d_load(void){
    const int height = 2;
    const int width = 2;
    matrix2d_t m = {0};
    float data[] = {1.f, 2.f, 3.f, 4.f};
    float* ptr = &data[0];

    matrix2d_load(&m, height, width, ptr);
    TEST_ASSERT_EQUAL_FLOAT(data[0], matrix2d_get_elem(&m, 0, 0));
    TEST_ASSERT_EQUAL_FLOAT(data[1], matrix2d_get_elem(&m, 0, 1));
    TEST_ASSERT_EQUAL_FLOAT(data[2], matrix2d_get_elem(&m, 1, 0));
    TEST_ASSERT_EQUAL_FLOAT(data[3], matrix2d_get_elem(&m, 1, 1));
}

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_matrix2d_get_elem_as_ref);
    RUN_TEST(test_matrix2d_get_elem_as_mut_ref);
    RUN_TEST(test_matrix2d_get_elem);
    RUN_TEST(test_matrix2d_set_elem);
    RUN_TEST(test_matrix3d_get_elem_as_ref);
    RUN_TEST(test_matrix3d_get_elem_as_mut_ref);
    RUN_TEST(test_matrix3d_get_elem);
    RUN_TEST(test_matrix3d_get_slice_as_mut_ref);
    RUN_TEST(test_common_matrix3d_reshape);
    RUN_TEST(test_common_matrix3d_reshape_2);
    RUN_TEST(test_matrix2d_load);
    
    int result = UNITY_END();

    return result;
}