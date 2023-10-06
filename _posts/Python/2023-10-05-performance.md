---
layout: single
title: "Python Performance Measurement"
categories: python
tags: [language, programming, python]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---



# 실행 속도

```python
import time

# 시작 시간 기록
start_time = time.time()

# 여기에 코드 실행

# 종료 시간 기록
end_time = time.time()

# 실행 시간 계산
execution_time = end_time - start_time

print(f"코드 실행 시간: {execution_time:.4f} 초")
```



# 사용 메모리 체크

## memory-profiler

- 설치: memory-profiler

```bash
pip install memory-profiler
```



- python code 작성

```python
from memory_profiler import profile

@profile
def your_function_name():
    # 여기에 함수 내용을 작성

if __name__ == "__main__":
    your_function_name()
```



- 실행

```bash
python -m memory_profiler your_script.py
```

## pytorch, pympler

- 설치: pympler

```bash
pip install pympler
```

- 코드

```python
from pympler import asizeof
def memory_usage(obj):
    return asizeof.asizeof(obj)

# 메모리 사용량 측정
x_memory_usage = memory_usage(x)
y_memory_usage = memory_usage(y)
w_memory_usage = memory_usage(w)
b_memory_usage = memory_usage(b)

print(f"Tensor x 메모리 사용량: {x_memory_usage} bytes")
print(f"Tensor y 메모리 사용량: {y_memory_usage} bytes")
print(f"Tensor w 메모리 사용량: {w_memory_usage} bytes")
print(f"Tensor b 메모리 사용량: {b_memory_usage} bytes")
```

