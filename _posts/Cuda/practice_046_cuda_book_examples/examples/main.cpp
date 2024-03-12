#include <iostream>
#include "utils.h"
#include <vector>
#include "example01.h"
#include "kernel.cuh"


int main(int argc, char* argv[]) {
//  int main() {

  // exampleVector();
  // printCudaDevicesInfo();
  MatMulUnder1024_Helper(argv);
  //std::cout << "Hello!\n";



  return 0;
}
