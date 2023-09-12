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



- There are a lot of interesting things you can do with tranformation matrixes. Later chapters will deal with this quite a bit, so I will spend some time here describing my notation scheme for nested transformations. As a non-trivial example I will include the database for an articulated human figure called Blobby Man. (Thjose of you who already know how to do human articulation, don't go away. There are some cute tricks here that are very useful.)

# The Mechanism

- This is an implementation of the well-known technique of nexted transformations. (Don't you just hate it when people call something "well known" and you have never heard of it? It soulds like they are showing off how many things they know. Well, admittedly we can't derive everything from scratch. But is sure would be nice to find a less smug way of saying so.)

- For those for whom this is not so well known, the basic idea behind nested transformations appears in several places, notably in Foley and van Dam and in Glassner. It is just an organization scheme to make it easier to deal with a hierarchy of accumulated transformations. It shows up in various software systems and has hardware implementations in the E&S Picture System or ths Silicon Graphics IRIS.

- Briefly, it works like this. We maintain a global 4 x 4 homogeneous coordinate transformation matrix called the current transformation, C, containing the transfomration from a primitive's definition space onto a desired location in screen space. I will assume a device-independent (buzz, buzz) screen space ranging from -1 to +1 in x and y and where z goes into the screen. This is a left-handed coordinate system.

- Each time a primitive is drawn, it is implicitly transformed by C. For example, the transformation of a (homogeneous) point is accomplished through simple matrix multiplication.

$$ [ x, y, z, w]_{scrn} = [x, y, z, w]_{defn}\mathbf{C} $$

- Other primitives can be transformed by some more complex arithmetic involving this matrix.

$$ \mathbf{C} $$ is typically the product of a perspective transformation and various rotations, translations, and scales. It is built up with a series of matrix multiplications by simpler matrices. Each multiplication premultiplies a new matrix into $$ \mathbf{C} $$.

$$ \mathbf{C} \leftarrow \mathbf{T}_{new} \mathbf{C} $$   

- Why in this order? Because a collection of objects, subobjects, subsubobjects, etc., is thought of as a tree-like structure. Drawing a picture of the scene is a top-down traversal of this tree. You encounter the more global of the transformations first and must miltiply them in as you see them. The transformations will therefore seem to be applied to the primitives in the reverse order to that in which they were multiplied into $$ \mathbf{C} $$. Another way you can think of it is that the transfomrations are applied in the same order stated, but that the coordinate system transfomrs along with the primitive as each elementary transformation is multiplied. At each node in the tree, of course, you can save and restore the current contents of $$ \mathbf{C} $$ on a stack.

# The Language

- The notation scheme I will use is not just a theoretical construct, it's what I actually use to do all my animitions. It admittedly has a few quirks, but I'm not goint to try to sanitize them because I want to be able to use databases I have actually tried out and to show listings that I know will work. I have purposely made each operation very elementary to make it easy to experiment with various combinations of transformations. Most reasonable graphics systems use something like this, so it shouldn't be too hard for you to translate my examples into your own language.

- Instructions for rendering a scene take the form of a list of commands and their parameters. These will be written here in TYPEWRITER type. All commands will have four or fewer letters. (The number 4 is used because of its ancient numerological significance.) Parameters will be separated by commas, not blanks. (Old-time FORTRAN programmers don't even see blanks, let alone use them as delimiters.) Don't complain, just be glad I'm not using O-Language (maybe I'll tell you about that sometime).

# Basic Command Set

- These commands modify $$ \mathbf{C} $$ and pass primitives through it. Each modification command premultiplies some simple matrix into $$ \mathbf{C} $$. No other action is taken. The command descriptions below will explicitly show the matrices used.

## Translation

```fortran
TRAN x, y, z
```

- premultiplies $$ \mathbf{C} $$ by an elementray translation matrix.

$$ 
\mathbf{C} \leftarrow \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
x & y & z & 1
\end{bmatrix} 
\mathbf{C} 
$$ 

## Scaling

```fortran
SCAL sx, sy, sz
```

- premultiplies $$ \mathbf{C} $$ by an elementray scaling matrix.

$$ 
\mathbf{C} \leftarrow \begin{bmatrix}
sx & 0 & 0 & 0 \\ 
0 & sy & 0 & 0 \\ 
0 & 0 & sz & 0 \\ 
0 & 0 & 0 & 1
\end{bmatrix} 
\mathbf{C} 
$$ 

## Rotation

```fortran
ROT theta, j
```

- The j parameter is an integer from 1 to 3 specifying the coordinate axis (x, y, or z). The positive rotation directions is given via the Right-Hand Rule (if you are using a left-handed coordinate system) or the Left-Hand Rule (if you are using a right-handed coordinate system). This may sound strange, but it's how it's given in Newman and Sproull. It makes positive rotation go clockwise when viewing in the direction of a coordinate axis. For each matrix below, we precalculate

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


## Perspective

```fortran
PERS a, z_n, z_f
```

- This transformation combines a perspective distortion with a depth (z) transformation. The perspective assumes the eye in at the origin, looking down the +z axis. The field of view is given by the angle $$ \alpha $$

- The depth transformation is specified by two values - $$ z_n $$ (the location of the new clipping plane) and $$ z_f $$ (the location of the far clipping plane). The matrix transforms $$ z_n $$ to +0, and $$ z_f $$ to +1. I know that the traditional names for these planes are hither and yon, but for some reason I always get these words mixed up, so I call them near and far.

- Precalculate the following quantities (note that far clipping can be effectively disabled by setting $$ z_f $$, which makes $$ \mathbf{Q}  = s $$ ).

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


## Orientation

```fortran
ORIE a, b, c, d, e, f, p, q, r
```

- Sometimes it's usefult to specify the rotation (orientation) portion of the transformation explicitly. There is nothing, though, to enforce it being a pure rotation, so it can be used for skew transformations.


$$ 
\mathbf{C} \leftarrow \begin{bmatrix}
a & d & p & 0 \\ 
b & e & q & 0 \\ 
c & f & r & 0 \\ 
0 & 0 & 0 & 1
\end{bmatrix} 
\mathbf{C} 
$$ 

## Transformation Stack

```fortran
PUSH
POP
```

- These two commands push and pop $$ \mathbf{C} $$ on/off the stack