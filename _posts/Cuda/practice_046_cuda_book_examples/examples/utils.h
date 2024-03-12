#ifndef UTILS_H
#define UTILS_H

#include <chrono>
#include <string>
#include <iostream>


namespace utils
{
  class Timer {
  private:
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::duration<double, std::milli> duration;
    std::string taskName;

  public:
    Timer();
    ~Timer();
    void on(const std::string& name);
    void elapsed();
  };

}

#endif