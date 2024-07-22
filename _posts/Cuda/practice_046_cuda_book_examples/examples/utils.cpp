#include "utils.h"


// Class Timer

namespace utils
{
  Timer::Timer() : duration(0.0) {}

  Timer::~Timer() {}

  void Timer::on(const std::string& name)
  {
    start = std::chrono::high_resolution_clock::now();
    taskName = name;
  }

  void Timer::elapsed()
  {
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Elapsed time for " << taskName << ": " << duration.count() << " ms\n";
  }



}
