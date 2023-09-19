import math
import numpy as np
from PIL import Image, ImageDraw
# import pygame
# from pygame.locals import *
# from OpenGL.GL import *
# from OpenGL.GLUT import *
# from OpenGL.GLU import *

# 초기화
def init():
    # 이미지 크기 및 배경 색상 설정
    width, height = 800, 800
    background_color = (255, 255, 255)
    # 이미지 생성 및 초기화
    image = Image.new("RGB", (width, height), background_color)
    return image, ImageDraw.Draw(image)

# 원근 투영 설정
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

# 이동
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


# 스케일링
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

# 회전
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


# Primitives
# 점 그리기
def draw_point(C, draw, position, color):
    x, y, z = position
    transformed_point = np.dot(C, np.array([x, y, z, 1]))[:3] # 3D 좌표를 원근 투영 변환
    x_2d, y_2d = transformed_point[0], transformed_point[1]  # 3D 좌표를 2D 좌표로 변환
    draw.point((x_2d, y_2d), fill=color) # 점 그리기

# 선 그리기
def draw_line(C, draw, start, end, color, width=1):
    draw.line([start, end], fill=color, width=width)

# 삼각형 그리기 (평면)
def draw_surface(draw, vertices, outline_color, fill_color=None):
    draw.polygon(vertices, outline=outline_color, fill=fill_color)


# 큐브 그리기
def draw_cube(C, draw, center, size, color):
    x, y, z = center
    half_size = size / 2

    # Cube vertices (육면체의 꼭지점, 8개)
    vertices = [
        (x - half_size, y - half_size, z - half_size),
        (x + half_size, y - half_size, z - half_size),
        (x + half_size, y + half_size, z - half_size),
        (x - half_size, y + half_size, z - half_size),
        (x - half_size, y - half_size, z + half_size),
        (x + half_size, y - half_size, z + half_size),
        (x + half_size, y + half_size, z + half_size),
        (x - half_size, y + half_size, z + half_size)
    ]

    # Define cube faces by specifying vertex indices
    faces = [
        [0, 1, 2, 3],  # Bottom
        [4, 5, 6, 7],  # Top
        [0, 1, 5, 4],  # Front
        [2, 3, 7, 6],  # Back
        [0, 3, 7, 4],  # Left
        [1, 2, 6, 5]   # Right
    ]

    for face in faces:
        # Convert vertex indices to vertex coordinates
        face_vertices = [vertices[i] for i in face]
        
        transformed_vertices = [np.dot(C, np.array([x, y, z, 1]))[:3] for x, y, z in face_vertices]
        # Convert 3D coordinates to 2D (for drawing)
        face_vertices_2d = [(v[0], v[1]) for v in face_vertices]
        
        # Draw the face with the specified color
        draw.polygon(face_vertices_2d, fill=color)


# 큐브 그리기
def draw_cube_edges(C, draw, center, size, color):

    print(f"C: ", C)

    x, y, z = center
    half_size = size / 2

    # Cube vertices (육면체의 꼭지점, 8개)
    vertices = [
        (x - half_size, y - half_size, z - half_size),
        (x + half_size, y - half_size, z - half_size),
        (x + half_size, y + half_size, z - half_size),
        (x - half_size, y + half_size, z - half_size),
        (x - half_size, y - half_size, z + half_size),
        (x + half_size, y - half_size, z + half_size),
        (x + half_size, y + half_size, z + half_size),
        (x - half_size, y + half_size, z + half_size)
    ]

    print(f"vertices: ", vertices)

    # Define cube edges by specifying vertex pairs
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges connecting top and bottom
    ]

    for edge in edges:
        # Convert vertex indices to vertex coordinates
        vertex1 = vertices[edge[0]]
        vertex2 = vertices[edge[1]]

        # Apply perspective transformation to edge vertices
        transformed_vertex1 = np.dot(C, np.array([vertex1[0], vertex1[1], vertex1[2], 1]))[:3]
        transformed_vertex2 = np.dot(C, np.array([vertex2[0], vertex2[1], vertex2[2], 1]))[:3]
        # transformed_vertices = [np.dot(C, np.array([v[0], v[1], v[2], 1]))[:3] for v in vertices]

        # Convert 3D coordinates to 2D (for drawing)
        vertex1_2d = (transformed_vertex1[0], transformed_vertex1[1])
        vertex2_2d = (transformed_vertex2[0], transformed_vertex2[1])

        # Draw the edge with the specified color
        draw.line([vertex1_2d, vertex2_2d], fill=color, width=2)




# 변환 행렬을 스택에 저장하기 위한 리스트
transformation_stack = []

# Transformation Stack: PUSH
# PUSH 명령어: 현재 변환 행렬을 스택에 저장
def push_matrix(matrix):
    global current_matrix
    transformation_stack.append([row[:] for row in current_matrix])

# Transformation Stack: POP
# POP 명령어: 스택에서 가장 최근에 저장한 변환 행렬을 꺼내와서 현재의 변환 행렬로 설정
def pop_matrix():
    global current_matrix
    if len(transformation_stack) > 0:
        current_matrix = [row[:] for row in transformation_stack.pop()]
    else:
        print("Stack is empty. Cannot POP.")

# # 그래픽 명령 수행 함수
# def perform_graphics_commands():
#     initialize()
#     perspective(45, 6.2, 11.8)
#     translate(0, -1.41, 9)
#     rotate(-80, 'x')
#     rotate(48, 'z')
#     scale(1, 1, -1)
#     draw_primitive('GPLANE')
#     push_matrix()
#     translate(0, 0, 1)
#     rotate(20, 'z')
#     draw_primitive('CUBE')
#     pop_matrix()
#     push_matrix()
#     scale(0.3, 0.4, 0.5)
#     translate(-5, -3.8, 1)
#     draw_primitive('CUBE')
#     pop_matrix()
# 
# 그래픽 명령 수행
# perform_graphics_commands()


# 메인 함수
def main():
    image, draw = init()
    C = np.identity(4)  # 초기 변환 행렬
    draw_cube_edges(C, draw, (400, 400, 150), 100, (255, 0, 0)) # draw_cube(draw, center, size, color, C)

    # C = perspective(C, 90, 0.1, 50.0)  # 원근 투영 설정, 수정 요망!
    C = translate(C, -100, -100, -200)
    # 그래픽 프리미티브 그리기 예제
    # draw_point(C, draw, (200, 200, 100), (0, 255, 0))
    # draw_line(draw, (50, 50), (350, 50), (0, 0, 255), width=2)
    # draw_surface(draw, [(100, 100), (50, 150), (150, 150)], (0, 255, 0))
    # draw_cube(draw, (200, 200, 150), 100, (255, 0, 255), C) # draw_cube(draw, center, size, color, C)
    draw_cube_edges(C, draw, (200, 200, 150), 100, (255, 0, 255)) # draw_cube(draw, center, size, color, C)

    # 이미지 표시
    # image = image.crop((-400, -400, 400, 400)) # left, top, right, bottom
    image.show()

if __name__ == "__main__":
    main()  # 메인 함수 실행