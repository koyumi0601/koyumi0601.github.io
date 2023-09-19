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
# def draw_cube(x, y, z):
#     glPushMatrix()  # 현재 변환 행렬을 스택에 저장
#     glTranslatef(x, y, z)  # 이동 변환 적용
#     glutSolidCube(1.0)  # 큐브 그리기
#     glPopMatrix()  # 스택에서 이전 변환 행렬을 복원

# 그리기 함수, edge 표시
def draw_cube(x, y, z):
    glPushMatrix()
    glTranslatef(x, y, z)
    size = 0.5  # 큐브의 반만큼 크기 설정
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

    glPopMatrix()


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