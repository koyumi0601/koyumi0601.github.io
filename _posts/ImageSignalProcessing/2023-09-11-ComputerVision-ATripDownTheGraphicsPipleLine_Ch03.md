---
layout: single
title: "A Trip Down The Graphics PipleLine, Chapter 03. Nested Transformations and Blobby Man"
categories: imagesignalprocessing
tags: [Image Signal Processing, A Trip Down The Graphics PipleLine]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*A Trip Down The Graphics PipleLine, Jim Blinn's Corner*



- There are a lot of interesting things you can do with transformation matrices. Later chapters will deal with this quite a bit, so I will spend some time here describing my notation scheme for nested transformations. As a non-trivial example I will include the database for an articulated human figure called Blobby Man. (Those of you who already know how to do human articulation, don't go away. There are some cute tricks here that are very useful.)

- 변환 행렬로 할 수 있는 흥미로운 작업이 많이 있습니다. 나중 장에서는 이에 대해 꽤 다루게 될 것이므로 여기에서 중첩된 변환에 대한 나의 표기 체계를 설명하는 데 시간을 할애하겠습니다. 비교적 복잡한 예로, "Blobby Man"이라고 불리는 관절 인간 모형의 데이터베이스를 포함하겠습니다. (인간 관절을 어떻게 다루는지 이미 알고 있는 분들도 떠나지 마세요. 여기에는 매우 유용한 재미있는 트릭이 몇 가지 있습니다.)


# The Mechanism

- This is an implementation of the well-known technique of nexted transformations. (Don't you just hate it when people call something "well known" and you have never heard of it? It soulds like they are showing off how many things they know. Well, admittedly we can't derive everything from scratch. But is sure would be nice to find a less smug way of saying so.)

- 이것은 잘 알려진 중첩 변환 기술의 구현입니다. (다른 사람들이 무엇인가를 "잘 알려진"이라고 부를 때 들으면 어떻게 생각하시나요? 그것은 그들이 얼마나 많은 것을 알고 있는지 자랑하고 있는 것 같아 불편하지 않으십니까? 솔직히 우리는 모든 것을 처음부터 유도할 수는 없습니다. 그러나 이렇게 말하는 덜 오만한 방법을 찾는 것은 확실히 좋을 것입니다.)

- For those for whom this is not so well known, the basic idea behind nested transformations appears in several places, notably in Foley and van Dam and in Glassner. It is just an organization scheme to make it easier to deal with a hierarchy of accumulated transformations. It shows up in various software systems and has hardware implementations in the E&S Picture System or the Silicon Graphics IRIS.

- 이것이 잘 알려진 것이 아닌 사람들에게는, 중첩 변환의 기본 아이디어는 여러 곳에서 나타납니다. 특히 Foley와 van Dam, 그리고 Glassner의 책에서 볼 수 있습니다. 이것은 누적된 변환의 계층 구조를 다루기 쉽게 만들기 위한 조직 체계일 뿐입니다. 이것은 다양한 소프트웨어 시스템에서 나타나며, E&S Picture System이나 Silicon Graphics IRIS와 같은 하드웨어 구현도 있습니다.

- Briefly, it works like this. We maintain a global 4 x 4 homogeneous coordinate transformation matrix called the current transformation, C, containing the transformation from a primitive's definition space onto a desired location in screen space. I will assume a device-independent (buzz, buzz) screen space ranging from -1 to +1 in x and y and where z goes into the screen. This is a left-handed coordinate system.

- 간단히 말하면, 이것은 다음과 같이 작동합니다. 우리는 현재 변환인 C라고 불리는 전역 4 x 4 동차 좌표 변환 행렬을 유지합니다. 이 행렬은 기본 정의 공간에서 화면 공간의 원하는 위치로의 변환을 포함합니다. 저는 장치 독립적인 (버릅이, 버릅이) 화면 공간을 가정합니다. 이 화면 공간은 x와 y에서 -1에서 +1로 범위가 지정되며, z는 화면 안쪽으로 향합니다. 이것은 왼손 좌표 시스템입니다.

> **추가 해설**
<br> **동차 좌표**
<br> 기하학적 객체(점, 벡터 등)를 표현하기 위해 추가적인 차원을 도입한 좌표 시스템
<br>
<br> **장점**
<br> 다양한 변환의 통합: 이동(translation), 회전(rotation), 크기 조정(scaling) 등 다양한 변환을 하나의 행렬 곱셈으로 표현할 수 있습니다.
<br> 편의성과 일관성: 여러 가지 변환이 동일한 형태의 행렬 곱셈으로 표현될 수 있기 때문에, 연산이 단순하고 일관성이 있습니다.
<br> 프로젝션: 3D -> 2D 투영과 같은 복잡한 변환을 간단하게 표현할 수 있습니다.
<br> 변환의 결합: 동차 좌표를 사용하면, 다양한 변환을 하나의 행렬로 합칠 수 있어 계산이 효율적입니다.
<br> 예시: scale -> translation

$$
\mathbf{C} \leftarrow \begin{bmatrix}
s_x & 0 & 0 & 0 \\ 
0 & s_y & 0 & 0 \\ 
0 & 0 & s_z & 0 \\ 
x & y & z & 1
\end{bmatrix} 
\mathbf{C} 
$$



- Each time a primitive is drawn, it is implicitly transformed by C. For example, the transformation of a (homogeneous) point is accomplished through simple matrix multiplication.

- 각각의 프리미티브가 그려질 때마다, 그것은 암묵적으로 C에 의해 변환됩니다. 예를 들어, (동차) 점의 변환은 간단한 행렬 곱셈을 통해 수행됩니다.

$$ [ x, y, z, w]_{scrn} = [x, y, z, w]_{defn}\mathbf{C} $$

> **추가 해설**
<br>
<br> 이 공식은 화면 공간으로의 변환을 나타내는데 사용됩니다. $$ [x, y, z, w]_{scrn} $$ 은 화면 공간의 좌표를 나타내고, $$ [x, y, z, w]_{defn} $$은 정의 공간의 좌표를 나타냅니다. 이 변환은 4x4 변환 행렬 C를 사용하여 정의됩니다. 이 공식을 사용하여 정의 공간에서 화면 공간으로 좌표를 변환할 수 있습니다.
<br>
<br> **정의 공간 (Definition Space):**
<br> 정의 공간은 그래픽 객체 또는 그림의 초기 위치와 크기를 정의하는 공간입니다.
<br>  이 공간에서는 그래픽 객체의 상대적인 크기, 위치 및 형태가 지정됩니다.
<br>  일반적으로 정의 공간은 그래픽 디자이너나 프로그래머가 그림을 설계하거나 초기화하는 단계에서 사용됩니다.
<br> 예를 들어, 정의 공간에서 원의 중심 좌표와 반지름을 지정할 수 있습니다.
<br> 
<br> **화면 공간 (Screen Space):**
<br> 화면 공간은 실제 화면에 표시되는 그래픽 객체의 위치와 크기를 나타내는 공간입니다.
<br> 이 공간은 화면의 픽셀 좌표 및 크기로 정의됩니다.
<br> 정의 공간에서 정의된 그래픽 객체가 실제 화면에 어떻게 배치되고 크기가 조절되는지를 나타냅니다.
<br> 예를 들어, 화면 공간에서 원이 실제 화면의 좌표와 크기로 표시됩니다.
<br> 
<br> 정의 공간은 그래픽 디자인 작업 및 모델링에 사용되며, 화면 공간은 실제로 화면에 그림을 렌더링하거나 표시할 때 사용됩니다. 변환 행렬과 같은 기술을 사용하여 정의 공간에서 화면 공간으로 객체를 변환하고 표시합니다.

- Other primitives can be transformed by some more complex arithmetic involving this matrix.

- 다른 기본 도형들은 이 행렬을 활용한 좀 더 복잡한 산술 연산을 통해 변환될 수 있습니다.

$$ \mathbf{C} $$ is typically the product of a perspective transformation and various rotations, translations, and scales. It is built up with a series of matrix multiplications by simpler matrices. Each multiplication premultiplies a new matrix into $$ \mathbf{C} $$.

- $$ \mathbf{C} $$는 일반적으로 원근 변환과 여러 회전, 이동 및 스케일 변환의 곱으로 구성됩니다. 이는 간단한 행렬들의 연속적인 행렬 곱셈으로 구성됩니다. 각 곱셈은 새로운 행렬을 $$ \mathbf{C} $$에 선행 곱셈(Pre-multiplication)합니다.

$$ \mathbf{C} \leftarrow \mathbf{T}_{new} \mathbf{C} $$   

- Why in this order? Because a collection of objects, subobjects, subsubobjects, etc., is thought of as a tree-like structure. Drawing a picture of the scene is a top-down traversal of this tree. You encounter the more global of the transformations first and must miltiply them in as you see them. The transformations will therefore seem to be applied to the primitives in the reverse order to that in which they were multiplied into $$ \mathbf{C} $$. Another way you can think of it is that the transfomrations are applied in the same order stated, but that the coordinate system transfomrs along with the primitive as each elementary transformation is multiplied. At each node in the tree, of course, you can save and restore the current contents of $$ \mathbf{C} $$ on a stack.

- 왜 이 순서일까요? 왜냐하면 물체, 하위 물체, 하위 하위 물체 등의 모음은 트리 구조로 생각됩니다. 장면의 그림을 그리는 것은 이 트리의 위에서 아래로 이동하는 것입니다. 먼저 더 전역적인 변환을 만나고 그것들을 볼 때마다 곱해져야 합니다. 따라서 변환은 $$ \mathbf{C} $$ 에 곱해진 순서와는 반대로 기본 도형에 적용되는 것처럼 보일 것입니다. 또 다른 생각 방식은 변환은 명시된 순서대로 적용되지만 각 기본 변환이 곱해질 때마다 좌표 시스템이 해당 도형과 함께 변환된다는 것입니다. 물론 트리의 각 노드에서 $$ \mathbf{C} $$ 의 현재 내용을 스택에 저장하고 복원할 수 있습니다.

> **추가 해설**
<br> **행렬에 적용되는 순서:**
<br> 원근 변환 (Perspective Transformation)
<br> 회전 (Rotation)
<br> 이동 (Translation)
<br> 스케일 변환 (Scaling)
<br> 
<br> **도형에 적용되는 순서:**
<br> 스케일 변환 (Scaling)
<br> 이동 (Translation)
<br> 회전 (Rotation)
<br> 원근 변환 (Perspective Transformation)
<br> 
<br> 행렬에 적용되는 순서와 도형에 적용되는 순서는 다를 수 있으며, 그 차이는 변환 행렬이 어떤 순서로 곱해지느냐에 따라 달라집니다. 변환 행렬의 곱셈은 순서에 민감하며, 순서를 바꾸면 결과가 달라질 수 있습니다.


> **추가 해설 2**
<br> **하위 물체와 하위 하위 물체의 트리 구조와 연결해서 설명**
<br> 이 트리 구조는 계층 구조로, 상위 객체가 하위 객체를 포함하는 방식으로 정의됩니다. 각 객체에는 자체 변환 행렬이 있고, 이 행렬은 상위 객체의 변환에 영향을 받습니다.
<br>
예를 들어, 상위 객체 A가 하위 객체 B를 포함하는 경우, A의 변환 행렬이 B의 변환에 영향을 미칩니다. 따라서 A의 변환 행렬을 먼저 적용하고 그 다음에 B의 변환 행렬을 적용해야 합니다. 이것이 왜 상위 변환 행렬을 먼저 곱하는 이유입니다.
<br>
트리 구조에서 상위 노드는 하위 노드에 대한 전역적인 변환을 나타내고, 하위 노드는 그 상위 노드에서 상대적인 변환을 나타냅니다. 이런 방식으로 행렬을 곱하면 전체 트리 구조의 객체가 올바르게 변환됩니다.

# The Language

- The notation scheme I will use is not just a theoretical construct, it's what I actually use to do all my animations. It admittedly has a few quirks, but I'm not going to try to sanitize them because I want to be able to use databases I have actually tried out and to show listings that I know will work. I have purposely made each operation very elementary to make it easy to experiment with various combinations of transformations. Most reasonable graphics systems use something like this, so it shouldn't be too hard for you to translate my examples into your own language.

- 내가 사용할 표기 체계는 이론적인 구조뿐만 아니라 실제로 모든 애니메이션을 만들 때 사용하는 것입니다. 솔직히 이 표기법은 약간의 독특한 점이 있지만, 나는 그것들을 정리하려고 하지 않을 것입니다. 왜냐하면 내가 실제로 시도해 본 데이터베이스를 사용하고 작동하는 것을 보여주기를 원하기 때문입니다. 나는 각각의 작업을 매우 기본적으로 만들어 실험하기 쉽도록 의도적으로 만들었습니다. 대부분의 합리적인 그래픽 시스템은 이와 유사한 것을 사용하므로 나의 예제를 자신의 언어로 번역하는 것은 그리 어렵지 않을 것입니다.

- Instructions for rendering a scene take the form of a list of commands and their parameters. These will be written here in TYPEWRITER type. All commands will have four or fewer letters. (The number 4 is used because of its ancient numerological significance.) Parameters will be separated by commas, not blanks. (Old-time FORTRAN programmers don't even see blanks, let alone use them as delimiters.) Don't complain, just be glad I'm not using O-Language (maybe I'll tell you about that sometime).

- 장면을 렌더링하기 위한 명령은 명령어와 그들의 매개변수로 이루어진 명령어 목록 형태로 제공됩니다. 이들은 TYPEWRITER 글꼴로 여기에 작성됩니다. 모든 명령어는 네 글자 이하로 구성됩니다. (네 자리 숫자는 그의 고대 수학적 의미 때문에 사용됩니다.) 매개변수는 공백이 아니라 쉼표로 구분됩니다. (옛날 FORTRAN 프로그래머들은 공백을 심지어 보지 않으며 구분자로 사용하지 않습니다.) 불평하지 마세요. 그냥 나 O 언어를 사용하지 않아서 기뻐하세요. (어쩌면 언젠가 그에 대해 이야기할지도 모르겠습니다.)

# Basic Command Set 기본 명령어 세트

- These commands modify $$ \mathbf{C} $$ and pass primitives through it. Each modification command premultiplies some simple matrix into $$ \mathbf{C} $$. No other action is taken. The command descriptions below will explicitly show the matrices used.

- 이러한 명령은 $$ \mathbf{C} $$를 수정하고 기본 도형을 그것을 통과시킵니다. 각 수정 명령은 어떤 간단한 행렬을 $$ \mathbf{C} $$에 선행 곱셈합니다. 다른 동작은 수행되지 않습니다. 아래의 명령어 설명에서는 사용된 행렬을 명시적으로 보여줍니다.

## Translation

```fortran
TRAN x, y, z
```

- premultiplies $$ \mathbf{C} $$ by an elementray translation matrix.

- C에 기본적인 이동 행렬을 선행 곱셈합니다.

$$
\mathbf{C} \leftarrow \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
x & y & z & 1
\end{bmatrix} 
\mathbf{C} 
$$

> **추가 코드**

```python
import numpy as np
# 현재 변환 행렬 C에 Translation을 적용하는 함수
def translate(C, x, y, z):
    translation_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [x, y, z, 1]
    ])
    # C와 Translation 행렬을 곱해 업데이트
    C = np.dot(C, translation_matrix)
    return C
```

> **추가 코드2**
<br> 컴퓨터 그래픽 처리를 위해 Python에서는 주로 Pygame, PyOpenGL, 또는 OpenGL을 사용하는 라이브러리와 함께 이동 변환(Translation)을 수행합니다. 

```bash
pip install PyOpenGL
pip install pygame
sudo apt-get install freeglut3-dev # OpenGL Utility Toolkit (GLUT) 라이브러리
# some NVIDIA Graphic Driver Installations
```

```python
import pygame  # pygame 라이브러리 임포트
from pygame.locals import *  # pygame.locals에서 모든 것을 임포트
from OpenGL.GL import *  # OpenGL.GL에서 모든 것을 임포트
from OpenGL.GLUT import *  # OpenGL.GLUT에서 모든 것을 임포트
from OpenGL.GLU import *  # OpenGL.GLU에서 모든 것을 임포트 (gluPerspective 함수를 사용하기 위해)

# 초기화 함수
def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # OpenGL 초기화 설정: 배경색을 검정으로 설정
    glEnable(GL_DEPTH_TEST)  # 깊이 테스트 활성화

# 그리기 함수
def draw_cube(x, y, z):
    glPushMatrix()  # 현재 변환 행렬을 스택에 저장
    glTranslatef(x, y, z)  # 이동 변환 적용
    glutSolidCube(1.0)  # 큐브 그리기
    glPopMatrix()  # 스택에서 이전 변환 행렬을 복원

# 메인 함수
def main():
    pygame.init()  # pygame 초기화
    display = (800, 600)  # 화면 크기 설정
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)  # OpenGL을 사용하는 화면 생성
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)  # 원근 투영 설정
    glTranslatef(0.0, 0.0, -5)  # 이동 변환 적용

    init()  # 초기화 함수 호출

    while True:
        for event in pygame.event.get():  # 이벤트 처리 루프
            if event.type == pygame.QUIT:  # 종료 이벤트 처리
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 화면 지우기

        # 이동 변환을 적용한 큐브 그리기
        draw_cube(0, 0, 0)  # 원래 위치
        draw_cube(2, 0, 0)  # X 축으로 이동
        draw_cube(0, 2, 0)  # Y 축으로 이동
        # draw_cube(0, 0, 2)  # Z 축으로 이동

        pygame.display.flip()  # 화면 업데이트
        pygame.time.wait(10)  # 10 밀리초 동안 대기

if __name__ == "__main__":
    glutInit()  # GLUT 초기화
    main()  # 메인 함수 실행
```

![Chater03_Translation_openGL]({{site.url}}/images/$(filename)/Chater03_Translation_openGL.png)



## Scaling

```fortran
SCAL sx, sy, sz
```

- premultiplies $$ \mathbf{C} $$ by an elementray scaling matrix.

- $$ \mathbf{C} $$를 기본 스케일링 행렬로 선행 곱셈합니다

$$
\mathbf{C} \leftarrow \begin{bmatrix}
sx & 0 & 0 & 0 \\ 
0 & sy & 0 & 0 \\ 
0 & 0 & sz & 0 \\ 
0 & 0 & 0 & 1
\end{bmatrix} 
\mathbf{C} 
$$

```python
import numpy as np

def scale(C, sx, sy, sz):
    scaling_matrix = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])
    # C와 Scaling 행렬을 곱해 업데이트
    C = np.dot(C, scaling_matrix)
    return C
```



> **추가 코드** OpenGL, Python



```python
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# 초기화 함수
def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # 배경색을 검정으로 설정
    glEnable(GL_DEPTH_TEST)  # 깊이 테스트 활성화

# 큐브 그리기 함수 (모서리 표시)
def draw_cube(x, y, z, angle, axis, scale):
    glPushMatrix()  # 현재 변환 행렬을 스택에 저장
    glTranslatef(x, y, z)  # 이동 변환 적용

    # 회전을 적용할 축에 따라 회전 매트릭스 생성
    if axis == 'x':
        glRotatef(angle, 1, 0, 0)  # X 축 주변으로 회전
    elif axis == 'y':
        glRotatef(angle, 0, 1, 0)  # Y 축 주변으로 회전
    elif axis == 'z':
        glRotatef(angle, 0, 0, 1)  # Z 축 주변으로 회전

    # 스케일 변환 적용
    glScalef(scale, scale, scale)

    size = 0.5  # 큐브 크기의 반
    vertices = [
        [-size, -size, -size],  # 0
        [size, -size, -size],   # 1
        [size, size, -size],    # 2
        [-size, size, -size],   # 3
        [-size, -size, size],   # 4
        [size, -size, size],    # 5
        [size, size, size],     # 6
        [-size, size, size]     # 7
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)  # 모서리 색상 (흰색)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

    glPopMatrix()  # 스택에서 이전 변환 행렬 복원

# 메인 함수
def main():
    pygame.init()  # pygame 초기화
    display = (800, 600)  # 화면 크기 설정
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)  # OpenGL을 사용하는 화면 생성
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)  # 원근 투영 설정
    glTranslatef(0.0, 0.0, -5)  # 이동 변환 적용

    init()  # 초기화 함수 호출

    angle = 0  # 회전 각도 초기화
    angle_increment = 1  # 회전 각도 증가량 설정
    scale = 1.0  # 스케일 초기화

    while True:
        for event in pygame.event.get():  # 이벤트 처리 루프
            if event.type == pygame.QUIT:  # 종료 이벤트 처리
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 화면 지우기

        # 이동 변환을 적용한 큐브 그리기
        # draw_cube(0, 0, 0, angle, 'x', scale)  # X 축으로 회전
        draw_cube(0, 0, 0, angle, 'x', 2)  # X 축으로 회전
        draw_cube(2, 0, 0, angle, 'y', scale)  # Y 축으로 회전
        draw_cube(0, 2, 0, angle, 'z', scale)  # Z 축으로 회전

        angle += angle_increment  # 회전 각도 증가

        pygame.display.flip()  # 화면 업데이트
        pygame.time.wait(10)  # 10 밀리초 동안 대기

if __name__ == "__main__":
    glutInit()  # GLUT 초기화
    main()  # 메인 함수 실행
```





![Chater03_Scale_openGL]({{site.url}}/images/$(filename)/Chater03_Scale_openGL.png)





## Rotation

```fortran
ROT theta, j
```

- The j parameter is an integer from 1 to 3 specifying the coordinate axis (x, y, or z). The positive rotation directions is given via the Right-Hand Rule (if you are using a left-handed coordinate system) or the Left-Hand Rule (if you are using a right-handed coordinate system). This may sound strange, but it's how it's given in Newman and Sproull. It makes positive rotation go clockwise when viewing in the direction of a coordinate axis. For each matrix below, we precalculate

- j 매개 변수는 좌표 축 (x, y 또는 z)을 지정하는 1에서 3까지의 정수입니다. 양의 회전 방향은 오른손 규칙 (왼손 좌표 시스템을 사용하는 경우) 또는 왼손 규칙 (오른손 좌표 시스템을 사용하는 경우)을 통해 제공됩니다. 이것은 이상하게 들릴 수 있지만, Newman과 Sproull에서 설명된 방식입니다. 이 방법은 좌표 축의 방향을 향해 볼 때 양의 회전이 시계 방향으로 이동하도록 만듭니다. 아래의 각 행렬에 대해 미리 계산합니다.

> **추가 해설**
<br> **왼손 좌표 시스템:**
<br> X 축: 오른쪽
<br> Y 축: 위
<br> Z 축: 화면을 향함
<br> 양의 회전: 시계 반대 방향
<br> 주로 컴퓨터 그래픽스 및 CAD 시스템에서 사용됩니다.
<br> 많은 3D 그래픽 소프트웨어 및 라이브러리에서 기본 좌표 시스템으로 사용됩니다.
<br> 
<br> **오른손 좌표 시스템:**
<br> X 축: 오른쪽
<br> Y 축: 위
<br> Z 축: 화면에서 밖으로 향함
<br> 양의 회전: 시계 방향
<br> 주로 공학 및 물리학에서 사용
<br> 일부 소프트웨어 및 시스템에서는 오른손 좌표 시스템을 사용합니다.


$$ s = sin \theta $$

$$ c = cos \theta $$

- The matrices are then

- j = 1 (x axis)

$$
\mathbf{C} \leftarrow \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & c & -s & 0 \\ 
0 & s & c & 0 \\ 
0 & 0 & 0 & 1
\end{bmatrix} 
\mathbf{C} 
$$



- j = 2 (y axis)

$$
\mathbf{C} \leftarrow \begin{bmatrix}
c & 0 & s & 0 \\ 
0 & 1 & 0 & 0 \\ 
-s & 0 & c & 0 \\ 
0 & 0 & 0 & 1
\end{bmatrix} 
\mathbf{C} 
$$


- j = 3 (z axis)

$$
\mathbf{C} \leftarrow \begin{bmatrix}
c & -s & 0 & 0 \\ 
s & c & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 1
\end{bmatrix} 
\mathbf{C} 
$$

```python
import numpy as np
import math

# 현재 변환 행렬 C에 회전을 적용하는 함수
def rotate(C, theta, j):
    # 라디안 단위로 각도를 변환
    theta = math.radians(theta)
    s = math.sin(theta)
    c = math.cos(theta)

    # 회전 축에 따라 회전 행렬 생성
    if j == 1:  # X 축 회전
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])
    elif j == 2:  # Y 축 회전
        rotation_matrix = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])
    elif j == 3:  # Z 축 회전
        rotation_matrix = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("j 매개 변수는 1, 2, 3 중 하나여야 합니다 (X, Y, Z 축 선택)")

    # C와 회전 행렬을 곱해 업데이트
    C = np.dot(C, rotation_matrix)
    return C
```



> **추가 코드** OpenGL, Python

```python
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# 초기화 함수
def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # 배경색을 검정으로 설정
    glEnable(GL_DEPTH_TEST)  # 깊이 테스트 활성화

# 큐브 그리기 함수 (모서리 표시)
def draw_cube(x, y, z, angle, axis):
    glPushMatrix()  # 현재 변환 행렬을 스택에 저장
    glTranslatef(x, y, z)  # 이동 변환 적용
    
    # 회전을 적용할 축에 따라 회전 매트릭스 생성
    if axis == 'x':
        glRotatef(angle, 1, 0, 0)  # X 축 주변으로 회전
    elif axis == 'y':
        glRotatef(angle, 0, 1, 0)  # Y 축 주변으로 회전
    elif axis == 'z':
        glRotatef(angle, 0, 0, 1)  # Z 축 주변으로 회전
    
    size = 0.5  # 큐브 크기의 반
    vertices = [
        [-size, -size, -size],  # 0
        [size, -size, -size],   # 1
        [size, size, -size],    # 2
        [-size, size, -size],   # 3
        [-size, -size, size],   # 4
        [size, -size, size],    # 5
        [size, size, size],     # 6
        [-size, size, size]     # 7
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)  # 모서리 색상 (흰색)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

    glPopMatrix()  # 스택에서 이전 변환 행렬 복원

# 메인 함수
def main():
    pygame.init()  # pygame 초기화
    display = (800, 600)  # 화면 크기 설정
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)  # OpenGL을 사용하는 화면 생성
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)  # 원근 투영 설정
    glTranslatef(0.0, 0.0, -5)  # 이동 변환 적용

    init()  # 초기화 함수 호출

    angle = 0  # 회전 각도 초기화
    angle_increment = 1  # 회전 각도 증가량 설정

    while True:
        for event in pygame.event.get():  # 이벤트 처리 루프
            if event.type == pygame.QUIT:  # 종료 이벤트 처리
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # 화면 지우기

        # 이동 변환을 적용한 큐브 그리기
        draw_cube(0, 0, 0, angle, 'x')  # X 축으로 회전
        draw_cube(2, 0, 0, angle, 'y')  # Y 축으로 회전
        draw_cube(0, 2, 0, angle, 'z')  # Z 축으로 회전

        angle += angle_increment  # 회전 각도 증가

        pygame.display.flip()  # 화면 업데이트
        pygame.time.wait(10)  # 10 밀리초 동안 대기

if __name__ == "__main__":
    glutInit()  # GLUT 초기화
    main()  # 메인 함수 실행
```






![Chater03_Rotation_openGL]({{site.url}}/images/$(filename)/Chater03_Rotation_openGL.png)



## Perspective 원근 투영

```fortran
PERS a, z_n, z_f
```

- This transformation combines a perspective distortion with a depth (z) transformation. The perspective assumes the eye in at the origin, looking down the +z axis. The field of view is given by the angle $$ \alpha $$

- 이 변환은 원근 왜곡(perspective distortion)과 깊이(z) 변환을 결합합니다. 원근은 원점에 있는 눈이 +z 축을 바라보는 것으로 가정합니다. 시야각은 각도 $$ \alpha $$에 의해 결정됩니다.

- The depth transformation is specified by two values - $$ z_n $$ (the location of the new clipping plane) and $$ z_f $$ (the location of the far clipping plane). The matrix transforms $$ z_n $$ to +0, and $$ z_f $$ to +1. I know that the traditional names for these planes are hither and yon, but for some reason I always get these words mixed up, so I call them near and far.

- 깊이 변환은 두 가지 값으로 지정됩니다 - $$ z_n $$ (새로운 클리핑 평면의 위치)과 $$ z_f $$ (원근투영의 멀리있는 클리핑 평면의 위치)입니다. 행렬은 $$ z_n $$ 을 +0으로 변환하고, $$ z_f $$ 를 +1로 변환합니다. 전통적인 용어로는 이러한 평면을 hither와 yon으로 부르지만, 어떤 이유로 항상 이 단어를 혼동해서 사용하지 않고, 대신 near와 far로 부릅니다.

- Precalculate the following quantities (note that far clipping can be effectively disabled by setting $$ z_f $$, which makes $$ \mathbf{Q}  = s $$ ).

- 다음의 양을 미리 계산하세요 (먼 클리핑은 $$ z_f $$ 를 설정하여 효과적으로 비활성화 할 수 있음에 유의하세요. 이로 인해 $$ \mathbf{Q}  = s $$ 가 됩니다).

$$ s = sin ( \frac{\alpha}{2}) $$

$$ c = cos ( \frac{\alpha}{2}) $$

$$ \mathbf{Q} = \frac{s}{1-z_n/z_f} $$

- The matrix is then

$$
\mathbf{C} \leftarrow \begin{bmatrix}
c & 0 & 0 & 0 \\ 
0 & c & 0 & 0 \\ 
0 & 0 & \mathbf{Q} & s \\ 
0 & 0 & -\mathbf{Q}z_n & 1
\end{bmatrix} 
\mathbf{C} 
$$


```python
import numpy as np
import math

def perspective(C, a, z_n, z_f):
    # 라디안 단위의 시야각을 계산
    alpha = math.radians(a)
    
    # 시야각에 따라 s와 c 계산
    s = math.sin(alpha / 2)
    c = math.cos(alpha / 2)
    
    # Q 계산 (먼 클리핑 비활성화 고려)
    if z_n == z_f:
        Q = s
    else:
        Q = s / (1 - z_n / z_f)
    
    # 원근 투영 변환 행렬 생성
    projection_matrix = np.array([
        [c, 0, 0, 0],
        [0, c, 0, 0],
        [0, 0, Q, s],
        [0, 0, -Q * z_n, 1]
    ])
    
    # C와 원근 투영 행렬을 곱해 업데이트
    C = np.dot(C, projection_matrix)
    return C
```



```python
gluPerspective(70, (display[0] / display[1]), 0.1, 50.0)  # 원근 투영 설정
    # fovy (field of view angle): 시야각입니다. 시야각은 카메라가 얼마나 넓은 각도로 화면을 볼 것인지를 나타냅니다. 일반적으로 각도로 표현되며, 45도가 일반적으로 사용되는 값입니다. 작은 각도는 좁은 시야를, 큰 각도는 넓은 시야를 의미합니다.
    # aspect (aspect ratio): 화면의 가로 세로 비율을 나타냅니다. 보통 display[0] / display[1]와 같이 화면의 가로 해상도를 화면의 세로 해상도로 나눈 값을 사용합니다. 이 값을 통해 화면의 비율을 고려하여 원근 투영 행렬을 생성합니다.
    # zNear (near clipping plane): 카메라에서 가까운 곳에서 잘려나가는 부분의 위치를 나타냅니다. 이 값은 카메라에서 물체가 얼마나 가까이 있어야 화면에 나타날지 결정합니다. 작은 양수 값 (보통 0.1 또는 그 이상)으로 설정합니다.
    # zFar (far clipping plane): 카메라에서 멀리 있는 곳에서 잘려나가는 부분의 위치를 나타냅니다. 이 값은 카메라에서 물체가 얼마나 멀리 있어야 화면에 나타날지 결정합니다. 무한대로 가까워지지 않도록 큰 양수 값 (보통 50.0 또는 그 이하)으로 설정합니다.
```



![Chater03_Perspective_openGL]({{site.url}}/images/$(filename)/Chater03_Perspective_openGL.png)



## Orientation

```fortran
ORIE a, b, c, d, e, f, p, q, r
```

- Sometimes it's useful to specify the rotation (orientation) portion of the transformation explicitly. There is nothing, though, to enforce it being a pure rotation, so it can be used for skew transformations.

- 때로는 변환의 회전(방향) 부분을 명시적으로 지정하는 것이 유용할 수 있습니다. 그러나 순수한 회전(rotation)이어야만 하는 것을 강제하는 요소는 없으므로, 이것은 기울기 변환(skew transformation)에도 사용될 수 있습니다.

$$
\mathbf{C} \leftarrow \begin{bmatrix}
a & d & p & 0 \\ 
b & e & q & 0 \\ 
c & f & r & 0 \\ 
0 & 0 & 0 & 1
\end{bmatrix} 
\mathbf{C} 
$$


```python
# 회전 및 방향 변환 행렬을 생성하는 함수
def orientation(a, b, c, d, e, f, p, q, r):
    orientation_matrix = np.array([
        [a, d, p, 0],
        [b, e, q, 0],
        [c, f, r, 0],
        [0, 0, 0, 1]
    ])
    return orientation_matrix
```


## Transformation Stack

```fortran
PUSH
POP
```

- These two commands push and pop $$ \mathbf{C} $$ on/off the stack.

- 이 두 명령어는 스택에 $$ \mathbf{C} $$를 push하고 pop하는 역할을 합니다.

> **추가 해설**
<br> 변환 스택을 관리하는 명령어. 현재 변환 행렬을 저장하고 관리하는데 사용
<br> 
<br> **PUSH** 현재의 변환 행렬 $$ \mathbf{C} $$를 스택에 저장
<br> **POP** 스택에서 가장 최근에 저장한 변환 행렬을 꺼내와서 현재의 변환 행렬 $$ \mathbf{C} $$로 설정
<br> 
<br> **사용예시**
<br> **복합 변환 관리**: 그래픽 객체에 여러 변환을 적용해야 할 때, 예를 들어 회전, 이동 및 스케일 변환을 동시에 사용해야 할 때, 각 변환을 스택에 저장하고 필요할 때마다 "POP"하여 이전 상태로 돌아갈 수 있습니다.
<br> **애니메이션**: 객체의 애니메이션을 만들 때, 프레임마다 다른 변환을 적용해야 할 수 있습니다. 각 프레임에 대한 변환 행렬을 스택에 저장하고 애니메이션 프레임을 전환할 때 "POP"하여 이전 프레임 상태로 돌아갈 수 있습니다.
<br> **렌더링 트리**: 복잡한 렌더링 구조에서 객체의 변환을 계층적으로 관리할 때 사용됩니다. 각 객체의 변환 정보를 스택에 저장하고 부모-자식 관계에 따라 스택을 관리하여 전체 렌더링 트리를 효율적으로 처리할 수 있습니다.
<br> **상태 저장 및 복원**: 그래픽 렌더링 상태를 저장하고 나중에 다시 복원해야 할 때 사용됩니다. 예를 들어, 사용자 상호 작용으로 인해 화면을 변경하고 이전 상태로 복원해야 하는 경우에 유용합니다.
<br> **트랜잭션**: 스택을 사용하여 변환 트랜잭션을 관리할 수 있습니다. 특정 작업의 일련의 변환을 그룹화하고 트랜잭션을 시작하고 종료하여 모든 변환이 적용되거나 적용되지 않도록 할 수 있습니다.


## Primitives

> **추가 해설**
<br> **Primitives**: 기본적인 도형이나 그래픽 요소
<br> 더 복잡한 그림을 만들기 위해 조합하거나 변형할 수 있는 기본적인 그래픽 빌딩 블록
<br> ex. Point, Line, Polygon, Circle, Rectangle, Surface, Implicit Surface, 3D model

```fortran
DRAW name
```

- A primitive could be a list of vector endpoints, point-and-polygons, implicit surfaces, cubic patches, blobbies, etc. This command means "pass the elements in primitive name (however it's defined) through $$ \mathbf{C} $$ and onto the screen."

- 원시(primitive)는 벡터 끝점 목록, 점 및 다각형, 암시적 표면, cubic 패치, blobbies 등이 될 수 있습니다. 이 명령은 "원시 이름에 정의된 방식대로 원시 요소를 $$ \mathbf{C} $$를 통과시켜 화면에 출력하라"는 뜻입니다.

# Example

- A typical scene will consist of an alternating sequence of $$ \mathbf{C} $$ - alteration commands and of primitive-drawing commands. At the beginning of the command list, $$ \mathbf{C} $$ is assumed to be initialized to the identity matrix.

- 일반적인 장면은 $$ \mathbf{C} $$ 변경 명령과 원시 그리기 명령의 번갈아 나오는 시퀀스로 구성됩니다. 명령 목록의 시작 부분에서는 $$ \mathbf{C} $$가 항등 행렬로 초기화되었다고 가정합니다.

- Here is a typical sequence of commands to draw a view of two cubes sitting on a grid plane. The primitive GPLANE consists of a grid of lines in the xy plane covering -2 to +2 along each axis, along with some labels and a tick-marked pole in the +z direction that is placed at y = 2. The primitive CUBE consists of a cube whose vertices have coordinates $$ [\pm 1, \pm 1, \pm 1] $$ - that is, it is centered at the origin and has edge length equal to 2. Notice the scale by 01 in z to convert from the right-handed system in which the scene is defined to the left-handed system in which it is rendered.

- 다음은 xy 평면에 있는 그리드 평면 위에 앉아 있는 두 개의 큐브를 그리는 일반적인 명령 순서입니다. 원시 GPLANE은 각 축을 따라 -2에서 +2까지 확장되는 xy 평면에서의 선 그리드로 구성되며, 일부 레이블과 y = 2에서 배치된 +z 방향의 틱 표시 막대를 포함합니다. 원시 CUBE는 정점이 $$ [\pm 1, \pm 1, \pm 1] $$ 좌표를 가지는 큐브로, 즉 원점을 중심으로 하고 모서리 길이가 2인 큐브입니다. 장면이 정의된 오른손 좌표 시스템에서 렌더링되는 왼손 좌표 시스템으로 변환하기 위해 z에서 01로 스케일을 변경하는 것에 주목하세요.

```fortran
PERS 45, 6.2, 11.8
TRAN 0, -1.41, 9
ROT -80, 1
ROT 48, 3
SCAL 1, 1, -1
DRAW GPLANE
PUSH
TRAN 0, 0, 1
ROT 20, 3
DRAW CUBE
POP
PUSH
SCAL .3, .4, .5
TRAN -5, -3.8, 1
DRAW CUBE
POP
```



- The results of executing these instructions appear in Figure 3.1.

- 이러한 명령을 실행한 결과는 그림 3.1에 나타납니다.

- Notice that the $$ z_n $$ and $$ z_f $$ variables are selected to bound the scene as closely as possible so that depth cueing will work. And, hey, it's called depth cueing, not depth queueing as I've seen some people write. (Depth queueing could perhaps be used to refer to a depth-priority rendering algorithm ... hmmm.)

- z$$ z_n $$ 및 $$ z_f $$ 변수가 장면을 가능한 한 근접하게 제한하도록 선택되어 있어 깊이 큐잉(depth cueing)이 작동할 것입니다. 그리고, 여기서 깊이 큐잉(depth cueing)이라고 불립니다. 어떤 사람들이 "깊이 큐잉"이라고 쓴 것을 본 적이 있는데, (깊이 우선 렌더링 알고리즘을 참조하는 데 사용될 수 있을 것 같습니다... 흠.)

# Possible Implementations 가능한 구현 방법

- Figure 3.1 Cubes on parade

- There are several ways you could perform the operations described by these lists of commands.

    - Translate them into explicit subroutine calls in some language implementation and compile them

    - Read them through a "filter" - type program that executes the commands as they are encountered. This is the way most of my rendering program work.

    - Read them into an "editor" - type program that tokenizes the commands into some interpreter data structure and reexecutes the sequence upon each frame update. This is the way my animation design program works.


- 이 명령 목록에 설명된 작업을 수행하는 방법은 여러 가지가 있습니다.

    - 어떤 언어 구현에서 명시적인 서브루틴 호출로 변환하고 컴파일하세요.

    - 그들을 "필터" 유형의 프로그램을 통해 읽어 명령이 발견되는 대로 실행하세요. 이것은 대부분의 렌더링 프로그램이 작동하는 방식입니다.

    - 이 명령을 "에디터" 유형의 프로그램으로 읽어 명령을 해석자 데이터 구조로 토큰화하고 각 프레임 업데이트마다 순차적으로 실행하세요. 이것은 내 애니메이션 디자인 프로그램이 작동하는 방식입니다.

# Advanced Commands 고급 명령

- The simple commands above can be implemented in about two pages of code. The enhancements below are a little more elaborate. The following constructions make sense only in the "editor" mode of operation.

- 위의 간단한 명령어는 약 두 페이지 정도의 코드로 구현할 수 있습니다. 아래의 개선 사항들은 조금 더 복잡합니다. 다음 구조들은 "편집기" 작업 모드에서만 의미가 있습니다.

## Parameters

- Any numeric parameter can be given a symbolic name. A symbol table will be maintained and the current numeric value of the symbol used when the instructions is executed. For example, our cube scene could be

- 모든 숫자 매개변수에는 기호 이름을 부여할 수 있습니다. 기호 테이블이 유지되며 명령이 실행될 때 기호의 현재 숫자 값이 사용됩니다. 예를 들어, 우리의 큐브 장면은 다음과 같이 표현될 수 있습니다.

```fortran
PERS FOV, ZN, ZF
TRAN XSCR, YSCR, ZSCR
ROT BACK, 1
ROT SPIN, 3
SCAL 1, 1, -1
DRAW GPLANE
PUSH
TRAN X1, Y1, Z1
ROT ANG, 3
DRAW CUBE
POP
PUSH
SCAL .3, .4, .5
TRAN -5, -3.8, Z1
DRAW CUBE
POP
```

- By setting the variables

- 변수를 설정함으로써

```fortran
FOV = 45      ZN = 6.2       ZF = 11.8
XSCR = 0      YSCR = 1.41    ZSCR = 9
BACK = -80    SPIN = 48
X1 = 0        Y1 = 0         Z1 = 1
ANG = 20
```

- and executing the command list, the same results would be generated. The same symbol can appear in more than one place, allowing a certain amount of constraint satisfaction.

- 변수를 설정하고 명령 목록을 실행하면 동일한 결과가 생성됩니다. 동일한 기호는 여러 위치에서 나타날 수 있으며, 일정한 양의 제약 조건을 제공합니다.

## Abbreviations 약어

- Each time a subobject is positioned relative to a containing object, the instructions usually look something like

- 각 서브 오브젝트가 포함 오브젝트와의 상대적인 위치에 배치될 때, 명령어는 보통 다음과 같이 보입니다.

```fortran
PUSH
# ..
# ..
various TRAN, ROT, SCAL commands
# ..
# ..
DRAW primitive
POP
```

- While explicit, the above notation is sometimes a bit spread out and hard to follow. This sort of thing happens so often that it's helpful to define an abbreviation for it. We do so by following the DRAW command (on the same line) by the list of transformation commands, separated by commas. An implied PUSH and POP encloses the transformation list and DRAW. Our cube scene now looks like

- 위의 표기법은 명시적이지만 때로는 조금 펼쳐져 있고 이해하기 어려울 수 있습니다. 이러한 유형의 상황은 자주 발생하기 때문에 이를 위한 약어를 정의하는 것이 도움이 됩니다. DRAW 명령 뒤에 (동일한 줄에) 쉼표로 구분된 변환 명령 목록을 추가하여 이를 수행합니다. 암시적인 PUSH와 POP이 변환 목록과 DRAW를 감싸게 됩니다. 이제 우리의 큐브 장면은 다음과 같이 보입니다.

```fortran
PERS FOV, ZN, ZF
TRAN XSCR, YSCR, ZSCR
ROT BACK, 1
ROT SPIN, 3
SCAL 1, 1, -1
DRAW GPLANE
DRAW CUBE, TRAN, X1, Y1, Z1, ROT, ANG, 3
DRAW CUBE, SCAL, .3, .4, .5, TRAN, -5, -3.8, Z1
```

## Subassembly Definitions 하위 어셈블리 정의

- These are essentially subroutines. A subassembly is declared and named by bracketing its contents by the commands

- 이들은 본질적으로 서브루틴입니다. 서브어셈블리는 다음 명령어로 선언 및 이름을 부여합니다.

```fortran
DEF name
# ..
any commands
# ..
```

- Once defined, a subassembly can be thought of as just another primitive. In fact, the "designer" of a list of commands should not know or care if the thing they are drawing is a primitive or a subassembly, so a subassembly is "called" by the same command as a primitive.

- 한 번 정의된 서브어셈블리는 단순히 다른 원시(primitive)와 같은 것으로 생각할 수 있습니다. 실제로 명령어 목록을 "설계자"가 그리는 대상이 원시인지 서브어셈블리인지 알거나 신경 쓸 필요가 없으므로, 서브어셈블리는 원시와 동일한 명령어로 "호출"됩니다.

```fortran
DRAW assy_name
```

- The subassembly calling and return process is completely independent of the matrix stack PUSH and POP process. Interpretation of commands begins at the built-in name WORLD.

- 서브어셈블리 호출 및 반환 프로세스는 행렬 스택 PUSH 및 POP 프로세스와 완전히 독립적입니다. 명령어의 해석은 내장된 이름 "WORLD"에서 시작됩니다.

- I typically organize my definitions so that WORLD contains only the viewing transformation, i.e., its rotations and transformations tell where the "camera" is and in which direction it is looking. My favorite all-purpose viewing transform is

- 일반적으로 나는 WORLD에는 보는 변환(viewing transformation)만 포함되도록 정의를 구성합니다. 즉, 회전 및 변환은 "카메라"가 어디에 있고 어느 방향을 보고 있는지를 나타냅니다. 제가 가장 선호하는 범용적인 보는 변환은 다음과 같습니다.

```fortran
DEF WORLD
PERS FOV, ZN, ZF
TRAN XSCR, YSCR, ZSCR
ROT BACK, 1
ROT SPIN, 3
ROT TINT, 1
TRAN -XLOOK, -YLOOK, -ZLOOK
SCAL 1, 1, -1
DRAW SCENE
```

- The variable XLOOK, YLOOK, and ZLOOK determine the "look-at" point. BACK, SPIN, and TILT trumble the scene about this point. Then XSCR, YSCR, and ZSCR position the "look-at" point on the screen. XSCR and YSCR might very well be zero, but ZSCR needs to be some positive distance to move the scene away from the eye.

- The variable XLOOK, YLOOK, and ZLOOK determine the "look-at" point. BACK, SPIN, and TILT trumble the scene about this point. Then XSCR, YSCR, and ZSCR position the "look-at" point on the screen. XSCR and YSCR might very well be zero, but ZSCR needs to be some positive distance to move the scene away from the eye.

- The assembly SCENE contains the contents of the scene and can be designed independently of how it is being viewed. Our cube scene again:

- 어셈블리 SCENE은 장면의 내용을 포함하며, 어떻게 보이는지와는 독립적으로 디자인할 수 있습니다. 다시 말해, 우리의 큐브 장면은 다음과 같습니다:

```fortran
DEF SCENE
DRAW GPLANE
DRAW CUBE, TRAN, X1, Y1, Z1, ROT, ANG, 3
DRAW CUBE, SCAL, .3, .4, .5, TRAN, -5, -3.8, Z1
```

# Blobby Man 

- A few years ago I made a short animation of a human figure called Blobby Man to illustrate a new surface modeling technique. Leaving aside issues of modeling, the figure itself is an interesting example of nested transformations. I have, in fact, used it as a homework assignment for my computer graphics class. (Gee, I guess I can't do that any more.)

- 몇 년 전에 저는 새로운 표면 모델링 기술을 설명하기 위해 Blobby Man이라는 인간 형상의 짧은 애니메이션을 만들었습니다. 모델링 문제를 제외하고, 이 형상은 중첩된 변환의 흥미로운 예입니다. 실제로 제 컴퓨터 그래픽스 수업에서 숙제 과제로 사용한 적이 있습니다. (저랑은 더 이상 그런 일을 할 수 없겠군요.)

- Blobby Man's origin is in his stomach, and he stands with the z axis vertical. The only primitive element is a unit radius SPHERE centered at the origin. The parameterized variables are all rotation angles. Their usage is defined in Table 3.1.

- Blobby Man의 원점은 그의 위복부에 있으며, z 축은 수직으로 서 있습니다. 유일한 원시 요소는 원점을 중심으로 하는 반지름 1의 구체(SPHERE)입니다. 매개변수화된 변수들은 모두 회전 각도입니다. 이들의 사용법은 표 3.1에서 정의되어 있습니다.

- The WORLD is the standard one given above. SCENE looks like 

- WORLD는 위에서 제시한 표준 설정입니다. SCENE은 다음과 같이 보입니다.

```fortran
DEF SCENE
DRAW GPLANE
DRAW TORSO, TRAN, XM, YM, ZM, ROT, RZM, 3,
```

- The actual articulated parts are

- 실제로 조립된 부분은 다음과 같습니다.

```fortran
DEF TORSO
DRAW LEFTLEG, TRAN, -0.178, 0, 0
DRAW RGHTLEG, TRAN, 0.178, 0, 0
DRAW SPHERE, TRAN, 0, 0, 0.08, SCAL, 0.275, 0.152, 0.153,
DRAW BODY, ROT, EXTEN, 1, ROT, BTWIS, 2, ROT, ROT, 3,
```

- Table 3.1 Meanings of Blobby Man variables


| Variable       | Meaning                                                                                                 |
|----------------|---------------------------------------------------------------------------------------------------------|
| EXTEN          | Extension. A dancers' term for bending forwards and backwards (x axis)                                  |
| ROT            | Rotation. A dancers' term for rotating the body and shoulders left and right about the verical (z) axis |
| BTWIS          | Angle of body leaning left and right (y axis)                                                           |
| NOD            | Head nod                                                                                                |
| NECK           | Head shake                                                                                              |
| LHIP, RHIP     | Angular direction that the leg is kicked                                                                |
| LOUT, ROUT     | Angular distance that the leg is kicked                                                                 |
| LTWIS, RTWIS   | Angle the leg is twisted about its length                                                               |
| LKNEE, RKNEE   | Knee bend                                                                                               |
| LANKL, RANKL   | Ankle bend                                                                                              |
| LSID, RSID     | Arm rotation to side                                                                                    |
| LSHOU, RSHOU   | Arm rotation forwards and back                                                                          |
| LATWIS, RATWIS | Arm rotation about its own length                                                                       |
| LELBO, RELBO   | Elbow angle                                                                                             |

```fortran
DEF BODY
DRAW SPHERE, TRAN, 0, 0, 0.62, SCAL, 0.306, 0.21, 0.5,
DRAW SHOULDER, TRAN, 0, 0, 1, ROT, EXTEN, 1, ROT, BTWIS, 2, ROT, ROT, 3,
```

```fortran
DEF SHOULDER
DRAW SPHERE, SCAL, 0.45, 0.153, 0.12,
DRAW HEAD, TRAN, 0, 0, 0.153, ROT, NOD, 1, ROT, NECK, 3,
DRAW LEFTARM, TRAN, -0.45, 0, 0, ROT, LSID, 2, ROT, LSHOU, 1, ROT, LATWIS, 3,
DRAW RGHTARM, TRAN, 0.45, 0, 0, ROT, RSID, 2, ROT, RSHOU, 1, ROT, RATWIS, 3, 
```


```fortran
DEF LEFTLEG        DEF RGHTLEG
PUSH               PUSH
ROT LHIP, 3,       ROT RHIP, 3,
ROT LOUT, 2,       ROT ROUT, 2,
ROT -LHIP, 3,      ROT -RHIP, 3,
ROT LTWIS, 3,      ROT RTWIS, 3,
DRAW THIGH         DRAW THIGH
TRAN 0, 0, -0.85,  TRAN 0, 0, -0.84,
ROT LANKL, 1       ROT RANKL, 1
DRAW FOOT          DRAW FOOT
POP                POP
```

```fortran
DEF LEFTARM        DEF RGHTRAM
PUSH               PUSH
DRAW UPARM         DRAW UPARM
TRAN 0, 0, -0.55,  TRAN 0, 0, -0.55,
ROT LELBO, 1,      ROT RELBO, 1,
DRAW LOWARM        DRAW LOWARM
TRAN 0, 0, -0.5,   TRAN 0, 0, -0.5,
DRAW HAND          DRAW HAND
POP                POP
```

- Some primitive body parts are defined as translated and squashed spheres as follows:

- 일부 기본 몸통 부분은 다음과 같이 이동 및 압축된 구로 정의됩니다:

```fortran
DEF HEAD
DRAW SPHERE, TRAN, 0, 0, 0.4, SCAL, 0.2, 0.23, 0.3
DRAW SPHERE, TRAN, 0, -0.255, 0.42, SCAL, 0.035, 0.075, 0.035,
DRAW SPHERE, TRAN, 0, 0, 0.07, SCAL, 0.065, 0.065, 0.14
DRAW SPHERE, TRAN, 0, -0.162, 0.239, SCAL, -0.0533, 0.0508, 0.0506,
```


```fortran
DEF UPARM
DRAW SPHERE, TRAN, 0, 0, -0.275, SCAL, 0.09, 0.09, 0.275,
```

```fortran
DEF LOWARM
DRAW SPHERE, TRAN, 0, 0, -0.25, SCAL, 0.08, 0.08, 0.25,
```

```fortran
DEF HAND
DRAW SPHERE, TRAN, 0, 0, -0.116, SCAL, 0.052, 0.091, 0.155,
```

```fortran
DEF THIGH
DRAW SPHERE, TRAN, 0, 0, -0.425, SCAL, 0.141, 0.141, 0.425,
```

```fortran
DEF CALF
DRAW SPHERE, SCAL, 0.05, 0.05, 0.05,
DRAW SPHERE, TRAN, 0, 0, -0.425, SCAL, 0.1, 0.1, 0.425,
```

```fortran
DEF FOOT
DRAW SPHERE, SCAL, 0.05, 0.04, 0.04, 
DRAW SPHERE, TRAN, 0, 0.05, -0.05, SCAL, 0.04, 0.04, 0.04,
DRAW SPHERE, TRAN, 0, -0.15, -0.15 ROT, -10.1, SCAL, 0.08, 0.19, 0.05,
```

- A picture of the result appears in Figure 3.2 The viewing parameters are

- 결과물의 그림은 Figure 3.2에 나타납니다. 뷰 파라미터는 다음과 같습니다:

```fortran
ZN = 5.17     ZF = 10.7
XSCR = -.1    YSCR = -1.6    ZXCR = 7.9
BACK = -90    SPIN = -30     TILT = 0
XLOOK = 0     YLOOK = 0      ZLOOK = 0
XM = 0        YM = 0         ZM = 1.75
```

- All other angles are 0.

- 다른 모든 각도는 0이라고 합니다.

- A picture of the man gesturing is in Figure 3.3. The view is the same, but the body angles are

- 남자가 제스처를 취하는 모습은 그림 3.3에 나와 있습니다. 뷰는 동일하지만 체모의 각도는 다음과 같습니다.

```fortran
NOD = -25    NECK = 28
RHIP = 105   ROUT = 13    RTWIS = -86    RKNEE = -53
LHIP = 0     LOUT = 0     LTWIS = 0      LKNEE = 0
LSID = -45   LSHOU = 0    LATWIS = -90   LELBO = 90
RSID = 112   RSHOU = 40   RATWIS = -102  RELBO = 85
```

- There are several tricks in the model of Blobby Man that are especially notable.

- Blobby Man 모델의 몇 가지 트릭 중 특히 주목할 만한 것들이 있습니다.

## Cumulative Transformations 누적 변환

- It is not neccessary to POP a transformation just after it is used to DRAW something. Sometimes it is useful to continuously accumulate translations and rotations. For example, Blobby Man's leg could have looked like

- 해당 변환을 사용하여 무언가를 그린 직후에 변환을 POP할 필요는 없습니다. 때로는 연속적으로 이동 및 회전을 누적하는 것이 유용할 수 있습니다. 예를 들어, Blobby Man의 다리는 다음과 같이 보일 수 있습니다.

- Figure 3.3 Blobby Man waving 

- Blobby Man가 손을 흔들고 있는 모습

```fortran
DEF LLEG
DRAW THIGH
DRAW CALFETC, TRAN, 0, 0, -0.85,
ROT, LKNEE, 1
```

```fortran
DEF CALFETC
DRAW CALF
DRAW FOOT, TRAN, 0, 0, -0.84, ROT, LANKL, 1
```

- As long as there are no transformed objects after that last one, some of the nesting can be dispensed with, leaving...

- 마지막 변환 이후에 변환된 객체가 없다면, 일부 중첩을 제거할 수 있습니다. 이렇게 남겨집니다.

```fortran
DEF LLEG
PUSH
DRAW THIGH
TRAN 0, 0, -0.85
ROT LKNEE, 1
DRAW CALF
TRAN 0, 0, -0.84
ROT LANKL, 1
DRAW FOOT
POP
```


## Repeated Variables 반복된 변수

- The variables EXTEN, BTWIS, and ROT are used twice, once to flex the BOY relative to the TORSO and once to flex the SHOULDER relative to the BODY. This gives a minimal simulation of a flexible spine for the figure.

- 변수 EXTEN, BTWIS 및 ROT은 두 번 사용되며, 한 번은 TORSO에 대한 BOY를 구부리고, 다른 한 번은 BODY에 대한 SHOULDER를 구부립니다. 이렇게 하면 모델이 유연한 척추를 최소한 시뮬레이션합니다.

## Rotated Rotations 회전된 회전

- The transformation of the (left) leg relative to the torso contains the sequence

- (왼쪽) 다리의 상대적인 허리 회전(transformations)은 다음 순서로 진행됩니다.

```fortran
ROT LHIP, 3
ROT LOUT, 2
ROT -LHIP, 3
```

- This is something I'm especially proud of. It is a not-completely-obvious variation of a common technique-using simple transformations to build rotations or scalings about points other than the origin. For example, if you wanted to rotate a primitive about a point at coordinates (DX, DY), the commands would be 

- 이것은 저가 특히 자랑스러운 부분입니다. 일반적인 기술의 미묘한 변형 중 하나입니다. 원점 이외의 점을 중심으로 회전 또는 스케일링을 구축하기 위해 간단한 변환을 사용합니다. 예를 들어, 좌표 (DX, DY)에서 원시(primitive)를 회전시키려면 다음 명령어를 사용해야 합니다.

```fortran
TRAN DX, DY, 0
ROT ANGLE, 3
TRAN -DX, -DY, 0
```

- In other words, you translate the desired rotation center to the origin, rotate, and then translate the center back to where it used to be. (Remember that the transformations will be effectively carried out in sequence in the reverse order from that seen above.) The rotation sequence used for the leg enables us to rotate the leg about a rotated coordinate axis. The purpose of this is to make the foot always point forwards, no matter what LHIP and LOUT are. Figure 3.4 shows how this works. it is a top view of just the legs and hips, and the dark line shows the axis of rotation by the angle LOUT. A similar technique could have been used for the arm-shoulder joints, but I didn't happen to need that much flexibility in the animiation.

- 다시 말해, 원하는 회전 중심을 원점으로 이동시킨 다음 회전하고, 마지막으로 중심을 이전 위치로 다시 이동시킵니다. (변환은 위에서 본 순서의 반대로 효과적으로 시행됩니다.) 다리에 사용된 회전 순서는 회전된 좌표 축을 중심으로 다리를 회전시킬 수 있게 해줍니다. 이렇게 하는 목적은 발이 항상 앞쪽을 가리키도록 만드는 것입니다. LHIP와 LOUT이 무엇이든지 상관없이 발이 항상 앞쪽을 가리키도록 하는 것입니다. 그림 3.4는 이 작업 방식을 보여줍니다. 다리와 엉덩이의 윗면만 보여주며, 어두운 선이 각도 LOUT에 따른 회전 축을 나타냅니다. 팔과 어깨 관절에 대해서도 유사한 기술을 사용할 수 있었지만, 애니메이션에서 그만큼의 유연성이 필요하지 않았기 때문에 사용하지 않았습니다.

# Addendum 부록

- I received a letter from Nelson Max about this chapter. He pointed out that the rotation trick for making the foot always point forwards does not keep it exactly forwards (with an x component of 0). It still has some small sideways component. This is, of course, quite true. My intention was just to keep it approximately pointing forwards (with a negative y component). This works best for the expected range of values -90 < LOUT < 0 and -90 < LHIP < 90. All other rotation combinations I tried made it too easy to get the foot pointing completely backwards, amusing perhaps, but a real nuisance for animation.

- 저는 Nelson Max로부터 이 장에 관한 편지를 받았습니다. 그는 발이 항상 완전히 앞쪽을 가리키지 않고 약간 옆으로 향한다는 점을 지적했습니다. 물론 이 말은 맞습니다. 제 의도는 발이 대략적으로 앞쪽을 가리키도록 하는 것이었습니다(음의 y 성분을 가지고). 이는 -90 < LOUT < 0 및 -90 < LHIP < 90의 예상 범위에서 가장 잘 작동합니다. 저는 시도한 다른 모든 회전 조합에서 발이 완전히 뒤로 향하게 되어 애니메이션에 큰 방해가 되는 경우가 많아 그렇게 하지 않았습니다. 그것은 어떤 면에서 유머스럽기는 하지만 애니메이션에는 실제로 방해가 되는 문제입니다.