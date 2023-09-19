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