#include <Python.h>
#include <numpy/arrayobject.h>

int main(int argc, char* argv[]) {
    // Python 인터프리터 초기화
    Py_Initialize();

    // NumPy 초기화
    import_array();

    // NumPy 배열 생성 및 다루기 예시
    int nd = 2; // 배열의 차원 수
    npy_intp dims[2] = {2, 3}; // 배열의 모양 (2x3 배열)
    double data[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}; // 데이터

    // NumPy 배열 생성
    PyObject* pArray = PyArray_SimpleNewFromData(
        nd, dims, NPY_DOUBLE, reinterpret_cast<void*>(data));

    // Python 코드 실행 예시
    PyObject* pModule = PyImport_ImportModule("mymodule"); // 예제 모듈 이름
    if (pModule != nullptr) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "my_function"); // 모듈의 함수 이름
        if (PyCallable_Check(pFunc)) {
            PyObject* pArgs = PyTuple_Pack(1, pArray); // NumPy 배열을 인자로 전달
            PyObject* pValue = PyObject_CallObject(pFunc, pArgs); // 함수 호출
            Py_XDECREF(pArgs);

            if (pValue != nullptr) {
                // 결과 처리
                // ...

                Py_XDECREF(pValue);
            } else {
                PyErr_Print();
            }
        } else {
            PyErr_Print();
        }

        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
    } else {
        PyErr_Print();
    }

    // Python 인터프리터 종료
    Py_Finalize();

    return 0;
}