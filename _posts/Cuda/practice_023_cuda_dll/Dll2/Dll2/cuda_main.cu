#include <stdio.h>
#include <stdlib.h>
#include "cuda_dll.h"

int main(void)
{
  printf("main: calling wrapper()\n");
  wrapper(5);
  fflush(stdout);
  return EXIT_SUCCESS;
}