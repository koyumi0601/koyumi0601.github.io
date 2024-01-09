#include "ipp.h"
#include "stdio.h"

int main() {
    Ipp8u src1[8*4] = {8, 4, 2, 1, 0, 0, 0, 0,
                       8, 4, 2, 1, 0, 0, 0, 0,
                       8, 4, 2, 1, 0, 0, 0, 0,
                       8, 4, 2, 1, 0, 0, 0, 0};

    Ipp8u src2[8*4] = {4, 3, 2, 1, 0, 0, 0, 0,
                       4, 3, 2, 1, 0, 0, 0, 0,
                       4, 3, 2, 1, 0, 0, 0, 0,
                       4, 3, 2, 1, 0, 0, 0, 0};

    Ipp8u dst[8*4];
    IppiSize srcRoi = {4, 4};
    int scaleFactor = 1;

    IppStatus status = ippiAdd_8u_C1RSfs(src1, 8, src2, 8, dst, 4, srcRoi, scaleFactor);

    if (status == ippStsNoErr) {
        printf("Result:\n");
        for (int i = 0; i < srcRoi.height; i++) {
            for (int j = 0; j < srcRoi.width; j++) {
                printf("%d ", dst[i * srcRoi.width + j]);
            }
            printf("\n");
        }
    } else {
        printf("Error: %s\n", ippGetStatusString(status));
    }

    return 0;
}