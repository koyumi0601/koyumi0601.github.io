#include <ipp.h>
#include <stdio.h>

int main() {
    IppStatus status;
    int len = 4; // 배열의 길이
    Ipp32f src1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Ipp32f src2[] = {0.5f, 1.5f, 2.5f, 3.5f};
    Ipp32f result[len];

    // Intel IPP 초기화
    status = ippInit();
    if (status != ippStsNoErr) {
        printf("Error: IPP initialization failed\n");
        return 1;
    }

    // 두 배열 더하기
    status = ippsAdd_32f(src1, src2, result, len);

    if (status != ippStsNoErr) {
        printf("Error: Array addition failed\n");
        return 1;
    }

    // 결과 출력
    printf("더한 결과 배열: ");
    for (int i = 0; i < len; i++) {
        printf("%.2f ", result[i]);
    }
    printf("\n");

    return 0;
}
