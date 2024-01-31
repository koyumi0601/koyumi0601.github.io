
###
import numpy as np
import torch
# a = np.array([1, 2, 3, 4])
a = torch.tensor([1, 2, 3, 4])
print(a)
print(type(a))
print(a.dtype)
print(a.shape)
# b = np.array([1, 2, 3.1, 4])
b = torch.tensor([1, 2, 3.1, 4])
print(b.dtype) # 하나라도 실수면 자동으로 실수 타입


###
print('-----------------')
A = np.array([[1, 2],[3, 4]]) 
# A = np.array([[1, 2], [3, 4, 5]]) # 행렬이라서, 각 행에 해당하는 숫자의 개수가 같아야 함. Error
print(A)
print(A.shape) # (2, 2)
print(A.ndim) # 차원의 수, 2
print(A.size) # 전체 성분의 수, 4

print('-----------------')
A = torch.tensor([[1, 2],[3, 4]]) 
print(A)
print(A.shape) # torch.Size([2, 2])
print(A.ndim) # 차원의 수, 2
print(A.size) # <built-in method size of Tensor object at 0x000001E6DBE7A450>. 텐서의 크기(size)를 나타내는 것이 아니라, 텐서 객체의 내부 메서드인 size()를 호출한 결과, 단순히 PyTorch 텐서 객체의 문자열 표현, 이 표현은 해당 텐서 객체가 메모리에 저장된 위치와 관련된 것이며, 텐서의 크기 정보를 직접 제공하지 않음
print(A.size()) # torch.Size([2, 2])
print(A.numel()) # 전체 성분의 수, 4



print('-----------------')
print(np.zeros(5)) # [0. 0. 0. 0. 0.]
print(np.zeros_like(A)) # 2x2
print(np.ones(5)) # 5x1
print(np.zeros((3,3))) # 3x3, 튜플이어야 함
print(np.arange(3, 10, 2)) # min 이상, max 미만, increment
print(np.arange(0, 1, 0.1)) # 
print(np.linspace(0, 1, 10, endpoint=False)) # min 이상, max 이하, number of element, max 미만 옵션

print('-----------------')
print(torch.zeros(5)) # [0. 0. 0. 0. 0.]
print(torch.zeros_like(A)) # 2x2
print(torch.ones(5)) # 5x1
print(torch.zeros(3,3)) # 3x3, 튜플 아니어도 됨. 튜플이어도 됨
print(torch.arange(3, 10, 2)) # min, max, increment
print(torch.arange(0, 1, 0.1))
print(torch.linspace(0, 1, 10)) # min 이상, max 이하, number of element, max 미만 옵션 없음


print('-----------------')
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b # 벡터의 합
print(c)

print('-----------------')
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b # 벡터의 합
print(c)


print('-----------------')
A = np.array([[1, 2, 3], [1, 2, 3]])
B = np.array([[4, 5, 6], [1, 1, 1]])
C = A + B
D = A - B
print(C)
print(D)
print()
print(A*B) # 성분 곱. Hadamard product
print(A/B) # 성분 나눗셈
print(B**2) # 성분 제곱

print('-----------------')
A = torch.tensor([[1, 2, 3], [1, 2, 3]])
B = torch.tensor([[4, 5, 6], [1, 1, 1]])
C = A + B
D = A - B
print(C)
print(D)
print()
print(A*B) # 성분 곱. Hadamard product
print(A/B) # 성분 나눗셈
print(B**2) # 성분 제곱

print('-----------------')
A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 2], [3, 4]])
print(A*B)
print(A@B) # 행렬곱

print('-----------------')
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[1, 2], [3, 4]])
print(A*B)
print(A@B) # 행렬곱

# 인덱싱 슬라이스
print('-----------------')
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[0])
print(a[1])
print(a[-1])
print(a[1:4])
print(a[7:])
print(a[:7])
print(a[:])

print('-----------------')
a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[0])
print(a[1])
print(a[-1])
print(a[1:4]) # index 1 이상 index 4 미만
print(a[7:])
print(a[:7]) # index 7 미만
print(a[:])

print('-----------------')
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A[0]) # [1, 2, 3]
print(A[-1]) # [7, 8, 9]
print(A[:]) # all
print(A[0][2]) # 3
print(A[0, 2]) # 3
B = [[1, 2, 3, 4], [5, 6, 7, 8]] # List
print(B)
print(B[0][2]) # 3
# print(B[0, 2]) # error
print(A[1, :]) # [4, 5, 6]
print(A[1, 0:3:2]) # [4, 6] 기억해 둘 것
print(A[:, 2]) # [3, 6, 9] 기억해 둘 것
print(A[:][2]) # [7, 8, 9] 위와 동일하진 않음


print('-----------------')
A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(A[0]) # [1, 2, 3]
print(A[-1]) # [7, 8, 9]
print(A[:]) # all
print(A[0][2]) # 3
print(A[0, 2]) # 3
B = [[1, 2, 3, 4], [5, 6, 7, 8]] # List
print(B)
print(B[0][2]) # 3
# print(B[0, 2]) # error
print(A[1, :]) # [4, 5, 6]
print(A[1, 0:3:2]) # [4, 6] 기억해 둘 것
print(A[:, 2]) # [3, 6, 9] 기억해 둘 것
print(A[:][2]) # [7, 8, 9] 위와 동일하진 않음

print('-----------------')
A = np.array([ [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]] ])
print(A)
print(A.shape) # (2, 3, 4). 높은 차원이 앞에 위치해있다. 채널, 행, 열
print(A[0, 1, 2]) # 채널, 행, 열
a = np.array([[[1, 2, 3, 4]]])
print(a.shape)
print(np.array(1).shape) # 0 차원도 존재. 대괄호의 갯수가 차원의 수와 같다.

print('-----------------')
A = torch.tensor([ [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]] ])
print(A)
print(A.shape) # (2, 3, 4). 높은 차원이 앞에 위치해있다. 채널, 행, 열
print(A[0, 1, 2]) # 채널, 행, 열
a = np.array([[[1, 2, 3, 4]]])
print(a.shape)
print(np.array(1).shape) # 0 차원도 존재. 대괄호의 갯수가 차원의 수와 같다.


# boolean 인덱싱
print('-----------------')
a = [1, 2, 3, 4, 5, 3, 3]
print(a==3) # 논리 연산자 ==, 리스트에서는 전체가 같은지 비교한다, False
A = np.array([[1, 2, 3, 4], [5, 3, 7, 3]])
print(A==3) # np.array에서는 각 성분에 대해서 비교한다. [[False False True False] [False True False True]]
print(A[A==3]) # Boolean indexing
A[A==3] = 100 
print(A) # [[1, 2, 100, 4], [5, 100, 7, 100]]

A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
B = np.array([True, False, False, True]) 
print(A[B, :]) # [[1, 2], [7, 8]

b = np.array([1, 2, 3, 4])
print(b[[True, True, False, False]]) # list로도 indexing이 된다.

c = [1, 2, 3, 4]
# c[[True, True, False, False]] # error


print('-----------------')
a = [1, 2, 3, 4, 5, 3, 3]
print(a==3) # 논리 연산자 ==, 리스트에서는 전체가 같은지 비교한다, False
A = torch.tensor([[1, 2, 3, 4], [5, 3, 7, 3]])
print(A==3) # np.array에서는 각 성분에 대해서 비교한다. [[False False True False] [False True False True]]
print(A[A==3]) # Boolean indexing
A[A==3] = 100 
print(A) # [[1, 2, 100, 4], [5, 100, 7, 100]]

A = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]) # 기억해 둘 것. 
B = torch.tensor([True, False, False, True]) 
print(A[B, :]) # [[1, 2], [7, 8]

b = torch.tensor([1, 2, 3, 4])
print(b[[True, True, False, False]]) # list로도 indexing이 된다. 기억해 둘 것

c = [1, 2, 3, 4]
# c[[True, True, False, False]] # error


# array indexing
print('-----------------')
a = np.array([1, 2, 3, 4, 5])
A = a[2]
print(A) # 3
A = a[ np.array( [2, 3, 4] ) ] # [3, 4, 5]
print(A)
A = a[ np.array( [[2, 2, 2], [3, 3, 3]] ) ] # [[3, 3, 3], [4, 4, 4]]
print(A)
a = [1, 2, 3]
# a[ [1, 1, 1, 1, 2, 2, 2] ] # error
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a[0]) # [1, 2, 3]
A = a[ np.array( [[0, 1], [1, 1]] ) ] # [[2, 5]]
print(A.shape)
print(A)

# tensor indexing
print('-----------------')
a = torch.tensor([1, 2, 3, 4, 5])
A = a[2]
print(A) # 3
A = a[ torch.tensor( [2, 3, 4] ) ] 
print(A) # [3, 4, 5]
A = a[ torch.tensor( [[2, 2, 2], [3, 3, 3]] ) ] 
print(A) # [[3, 3, 3], [4, 4, 4]]
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a[0]) # [1, 2, 3]
A = a[ torch.tensor( [[0, 1], [1, 1]] ) ] 
# 이건 기억해 둘 것. 
# 요소 각각이 하나의 열을 지시 함. 
# 0번째 열의 모든 행 부터 1번째 열의 모든 행을 지시한 것. [1, 2, 3] [4, 5, 6] 차원이 늘어나는 것.
# segmentation 결과 그림 보여줄 때 사용
# 예를 들어, 강아지인 영역이 3으로 묶였을 때, 이 영역을 RGB로 mapping해야 하는 경우가 생김.
print(A.shape) # torch.Size([2, 2, 3])
print(A) # [ [[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [4, 5, 6]] ]

print('-----------------')
A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
print(A)
print(A.shape) # (4, 2)
# 1. A[행, 열]
print(A[0, 1])
# 2. A[ array(bool) ] => A와 같은 shape을 가지는 array형 bool이 어디에 True를 가지고 있냐
print(A[ np.array( [[False, True], [False, False], [False, False], [False, False]]) ]) # 2. array-bool인 경우. 기억해 둘 것
print(A[A==2])
# 3. A[몇 번째 값에 True가 있냐, 몇 번째 값에 True가 있냐]
print(A[ [True, False, False, True], [False, True]]) # list-bool인 경우. 기억해 둘 것.
# [행 중 선택][열 중 선택] [2, 8]
# 4. A[ array ] # 몇 번째 것을 어떻게 쌓을 거냐
print(A[ np.array([1, 1, 2, 2, 2]) ]) # [[3, 4], [3, 4], [5, 6], [5, 6], [5, 6]]

print('-----------------')
A = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
print(A)
print(A.shape) # (4, 2)
# 1. A[행, 열]
print(A[0, 1])
# 2. A[ array(bool) ] => A와 같은 shape을 가지는 array형 bool이 어디에 True를 가지고 있냐
print(A[ torch.tensor( [[False, True], [False, False], [False, False], [False, False]]) ]) # 2. array-bool인 경우. 기억해 둘 것
print(A[A==2])
# 3. A[몇 번째 값에 True가 있냐, 몇 번째 값에 True가 있냐]
print(A[ [True, False, False, True], [False, True]]) # list-bool인 경우. 기억해 둘 것.
# [행 중 선택][열 중 선택] [2, 8]
# 4. A[ array ] # 몇 번째 것을 어떻게 쌓을 거냐
print(A[ torch.tensor([1, 1, 2, 2, 2]) ]) # [[3, 4], [3, 4], [5, 6], [5, 6], [5, 6]]


# numpy의 여러 함수들
print('-----------------')
A = np.random.randn(3, 3) # -1~1
B = np.random.rand(3, 3) # 0~1
print(A)
print(B)
print(A[A[:, 0] < 0, :]) # 0번째 열이 음수인 경우, 모든 열

# pytorch의 여러 함수들
print('-----------------')
A = torch.randn(3, 3) # -1~1
B = torch.rand(3, 3) # 0~1
print(A)
print(B)
print(A[A[:, 0] < 0, :]) # 0번째 열이 음수인 경우, 모든 열. 데이터 뽑아낼 때

print('-----------------')
A = np.random.randn(3, 3)
print(A)
print(np.abs(A))
print(np.sqrt(np.abs(A)))
print(np.exp(A))
print(np.log(np.abs(A)))
print(np.log(np.exp(1)))
print(np.log10(10))
print(np.log2(2))
print(np.round(A))
print(np.round(A, 2))
print(np.floor(A))
print(np.ceil(A))

print('-----------------')
A = torch.randn(3, 3)
print(A)
print(torch.abs(A))
print(torch.sqrt(torch.abs(A)))
print(torch.exp(A)) # A = [[1, 2], [3, 4]]라면, [[e^1, e^2], [e^3, e^4]]. 각 요소에 대한 반복 처리라고 보면 됨.
print(torch.log(torch.abs(A)))
print(torch.log(torch.exp(torch.tensor(1)))) # input으로 tensor만 받음. int나 float 안됨
print(torch.log10(torch.tensor(10)))
print(torch.log2(torch.tensor(2)))
print(torch.round(A))
print(torch.round(A, decimals=2)) # argument 순서가 다른 듯
print(torch.floor(A))
print(torch.ceil(A))

print('-----------------')
print(np.sin(np.pi/6))
print(np.cos(np.pi/3))
print(np.tan(np.pi/4))
print(np.tanh(-10))

print('-----------------')
print(torch.sin(torch.tensor(torch.pi/6))) # torch.sin의 input argument가 tensor여야 함.
print(torch.sin(torch.pi/torch.tensor(6))) # tensor끼리 연산하면 type을 tensor로 바꿔줌.
# print(torch.sin(torch.pi/6)) # tensor/6 type은 flaot
print(type(torch.pi/6)) # tensor/6 type은 flaot
print(torch.cos(torch.pi/torch.tensor(3)))
print(torch.tan(torch.pi/torch.tensor(4)))
print(torch.tanh(torch.tensor(-10)))

print('-----------------')
print(np.nan) # not a number
print(np.log(-1)) # runtime warning. invalid input argument. 경고는 나오지만 nan 출력 됨.
print(np.isnan([1, 2, np.nan, 3, 4])) # [False False True False False] 출력은 boolean
print(np.isinf([1, 2, 3, 4, np.inf])) # [False False False False True]

print('-----------------')
print(torch.nan) # not a number
print(torch.log(torch.tensor(-1)))
print(np.isnan(torch.tensor([1, 2, torch.nan, 3, 4]))) # tensor([0, 0, 1, 0, 0] dtype=torch.uint8
print(np.isinf(torch.tensor([1, 2, 3, 4, torch.inf]))) # tensor([0, 0, 0, 0, 1] dtype=torch.uint8

print('-----------------')
A = np.random.randn(3, 4)
print(A)
print(np.max(A)) # 전체 최대값
print(np.max(A, axis=0)) # 각 열에 대한 최대값
print(np.max(A, axis=1)) # 각 행에 대한 최대값
print(np.min(A))
print(np.min(A, axis=0))
print(np.min(A, axis=1))
print(np.argmax(A)) # 전체 최대값의 위치. 즉, 인덱스. (n, m)같은 주소 아니고 전체 요소에 대한 몇 번째 요소인지 찾음. 1D array로 바꾼다고 보면 됨.
print(np.argmax(A, axis=0))
print(np.argmax(A, axis=1))
# max vs argmax
# -(x-1)^2
# max: 최대값, 0
# argmax: 최대값을 만드는 입력값, 1

print('-----------------')
A = torch.randn(3, 4)
print(A)
print(torch.max(A)) # 전체 최대값
# print(torch.max(A, axis=0)) # 각 열에 대한 최대값과 인덱스를 같이 리턴. axis, dim 둘 다 동작 함. 공식문서가 dim이므로 dim을 주로 사용.
print(torch.max(A, dim=0)) # 각 열에 대한 최대값과 인덱스를 같이 리턴
max_val, max_ind = torch.max(A, dim=0)
print(f'max_val: {max_val}, max_ind: {max_ind}')
print(torch.max(A, dim=1)) # 각 행에 대한 최대값과 인덱스를 같이 리턴
print(torch.min(A))
print(torch.min(A, dim=0))
print(torch.min(A, dim=1))
print(torch.argmax(A)) # 전체 최대값의 위치. 즉, 인덱스. (n, m)같은 주소 아니고 전체 요소에 대한 몇 번째 요소인지 찾음.
print(torch.argmax(A, dim=0))
print(torch.argmax(A, dim=1))

print('-----------------')
a = np.random.randn(6, 1)
print(a)
a_sorted = np.sort(a, axis=0) # 오름차순 정렬. 낮은 값 -> 높은 값
print(a_sorted)
print(np.flipud(a_sorted))

print('-----------------')
a = torch.randn(6, 1)
print(a)
a_sorted = torch.sort(a, dim=0) # 오름차순 정렬. 낮은 값 -> 높은 값, 인덱스도 함께 반환
# a_sorted = torch.sort(a) # dim을 주지 않으면, randn(6, 1)에서 마지막 dimension에 대해서 sort 수행하므로, 원하는 동작이 되지 않는다.
print(a_sorted) 
a_sorted_val, a_sorted_ind = torch.sort(a, dim=0) # flipud 입력 타입이 tensor여야해서, 따로 받음.
print(torch.flipud(a_sorted_val)) # flipup 입력 타입이 tensor여야 함.
print(torch.sort(a, dim=0, descending=True)) # 내림차순정렬
print(torch.sort(a, dim=0, descending=False)) # 오름차순정렬
print(torch.sort(a, dim=0)) # 오름차순정렬, default False

print(f'a.sort(dim=0): {a.sort(dim=0)}') # class의 함수를 쓰기도 함.
print(f'torch.max(a): {torch.max(a)}, a.max(): {a.max()}') # max도 class의 함수를 많이 씀. 코드의 간결성. torch.max()는 추가 작업이므로 약간의 오버헤드 발생. 인덱스도 리턴한다는 장점이 있으나 최대값만 필요한 경우에는 후자를 사용한다.


print('-----------------')
A = np.random.randn(3, 4)
print(A)
print(np.sum(A))
print(np.sum(A, axis=1))
print(np.sum(A, axis=1, keepdims=True)) # 예를 들어 shape [3,4] -> [3]으로 바뀔텐데, [3,1]로 유지하라는 옵션
print(np.mean(A))
print(np.mean(A, axis=1))
print(np.mean(A, axis=1, keepdims=True))
print(np.std(A))

print('-----------------')
A=torch.randint(1, 5, size=(12,)) # 1~5미만 랜덤 integer 12x1
print(A)
print(A.shape)
B=A.reshape(2,2,3)
print(B)
print(B.ndim)


print('-----------------')
# 1:37:11
a=torch.tensor([1, 2, 3])
b=torch.tensor([2, 2, 1])
print(torch.sum(a*b))

a=a.reshape(3, 1)
b=b.reshape(3, 1)
print(a.transpose(0, 1)@b) # transpose(dim1, dim2). 서로 바꿈. numpy는 a.transpose() 허용
print(a.permute(1, 0)@b) # transpose. dim1을 1에 배치, dim0을 0에 배치 이런 식.
print(a.T@b)

A=torch.randn(4, 3, 6)
# print(A.transpose(0, 2, 1).shape) # dim1, dim2 두 개만 받음. 여러번 수행해야 함.