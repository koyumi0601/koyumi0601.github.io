#include <GL/glut.h>

// Display callback function
void display() {
    glClear(GL_COLOR_BUFFER_BIT);  // Clear the color buffer
    // generate triangle
    glBegin(GL_TRIANGLES);  // Begin drawing triangles
    glColor3f(1.0f, 0.0f, 0.0f);  // Set color to red
    glVertex2f(0.0f, 1.0f);  // Vertex 1 (top)
    glColor3f(0.0f, 1.0f, 0.0f);  // Set color to green
    glVertex2f(-1.0f, -1.0f);  // Vertex 2 (bottom-left)
    glColor3f(0.0f, 0.0f, 1.0f);  // Set color to blue
    glVertex2f(1.0f, -1.0f);  // Vertex 3 (bottom-right)
    glEnd();  // End drawing triangles

    glFlush();  // Flush the OpenGL pipeline
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);  // Initialize GLUT
    glutCreateWindow("Simple OpenGL Example");  // Create a window
    glutDisplayFunc(display);  // Set the display callback function
    glutMainLoop();  // Enter the main loop
    return 0;
}