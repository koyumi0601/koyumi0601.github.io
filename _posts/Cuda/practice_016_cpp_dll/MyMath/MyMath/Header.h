#ifndef MYMATH_H_
#define MYMATH_H_

#include <memory>
#include <functional>
#include "Define.h"

#ifdef MYMATH_EXPORTS
#define MYMATH_MYDLL_DECLSPEC __declspec(dllexport)
#else
#define MYMATH_MYDLL_DECLSPEC __declspec(dllimport)
#endif

namespace PlusDLL
{
  class MYMATH_MYDLL_DECLSPEC MyPlus
  {
  private:

  public:
    MyPlus(void)
      ~MyPlus(void);
    int PracticePlus(int a, int b);
  };
}

#endif