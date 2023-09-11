---
layout: single
title: "A Trip Down The Graphics PipleLine, Chapter 01. How Many Ways Can You Draw a Circle?"
categories: imagesignalprocessing
tags: [Image Signal Processing, A Trip Down The Graphics PipleLine]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*A Trip Down The Graphics PipleLine, Jim Blinn's Corner*

- I like to collect things. When I was young I collected stamps; now I collect empty margarine tubs and algorithms for drawing circles. I will presume that the latter is of most interest to you, so in this chapter I will go through may album of circl-drawing algorithms.

- It's traditional at this point in any discussion of geometry to drag in the ancient Greeks and mention how they considered the circle the most perfect shape. Even though a circle is such an apparently simple shape, it is interesting to find how many essentially different algorithms you can find for drawing the Greeks' favorite curve.

- I will be very brief about some pretty obvious techniquest to leave space to play with the more interesting and subtile techniques, Note that many of these algorithms might be ridiculously inefficient but are included to pad the chapter. (OK, OK, they're included for completeness.)

- I'm not sure where I first heard of some of these. I will cite inventors where known, but let me just thank the world at large in case I've missed anybody.

- A word about the programming language used: I am not using any formal algorithm display language here. These algorithms are meant to be read by human beings, not computers, so the language is a mishmash of several programming constructs that I hope will be perfectly clear to you.

- The collection can be categorized by the two types of output, line endpoints or pixel coordinates. This comes from the general dichotomy of curve representation-parametric versus algebraic.

# Line Drawings

- First let's look at line output. All these algorithms will operate in floating point and generate a series of x,y points on a unit radius circle centered at the origin. You then play connect-the-dots.

## (1) Trigonometry

- Evaluate sin and cos at equally spaced angles.

```python
MOVE(1, 0)
FOR DEGREES = 1 to 360
    DARIANS = DEGREES * 2 * 3.14159/360.
    DRAW(COS(RADIANS), SIN(RADIANS))
```

- This has to evalute the two trig functions at each loop; ick.

## (2) Polynomial Approximation

- You can get a fair approximation to a circle by evaluating simple polynomial approximations to sin and cos. The first ones that come to mind are the Taylor series.

$$ cos \alpha \approx 1 - 1/2 \alpha^2 + \alpha/24 \alpha^4 $$

$$ sin \alpha \approx \alpha - 1/6\alpha^3 +1/120 \alpha^5 $$

- These require fairly high-order terms to get very close, partly because the Talor series just fits the position and several derivatives at one endpoint.

- A better approach is to fit lower-order polynomials to both desired endpoints and end slopes. This is effectively what is happening with various commonly used Bezier curves. For example, the four control points(1,0), (1, .552), (.552, 1), (0, 1) describe a good approximation to the upper-right quarter of a circle. You can get the other three quadrants by rotating the control points.

- When transformed to polynomial form, the first quadrant is

$$ x(t) = 1-1.344t^2 + 0.344t^3 $$

$$ y(t) = 1.656t - 0.312t^2 - 0.344t^3 $$

- with the parameter t goint from 0 to 1.

```python
MOVE(1, 0)
FOR T = 0 TO 1 BY .01
    X = 1 + T * T * (-1.344 + T * 0.344)
    Y = T * (1.656 - T * (0.312 + T * 0.344))
    DRAW(X, Y)
```

- This makes a pretty good circle. The maximum radius error is about .0004 at t = 0.2 and t = 0.8.


## (3) Forward Differences

- Polynomials can be evaluated quickly by the technique known as Forward Differences. Briefly, for the polynomial

$$ f(t) = f_0 +f_1 t + f_2 t^2 + f_3 t^3 $$

- if you start at t = 0 and increment by equal steps of size $$ \delta $$, the forward differences are

$$ \Delta f = f_1 \delta + f_2 \delta^2 + f_3 \delta^3 $$

$$ \Delta \Delta f = 2 f_2 \delta^2 + 6 f_3 \delta^3 $$

$$ \Delta \Delta \Delta f = 6 f_3 \delta^3 $$

- Then, for our polynomials stepping in units of 0.01,

```python
X = 1; DX = -.000134056; DDX = -.000266736; DDDX = .000002064
Y = 0; DY = .016528456; DDY = -.000064464; DDDY = -.000002064
MOVE(X, Y)
FOR I = 1 TO 100
    X = X + DX; DX = DX + DDX; DDX = DDX + DDDX
    Y = Y + DY; DY = DY + DDY; DDY = DDY + DDDY
    DRAW(X, Y)
```

- Trust me, I'm a doctor. If you don't belive it, look up Forward Differences in Newman and Sproull's book - I'm not goint to do all the work here.

- Notice the number of significant digits in the constants. It might seem that tha many digits would require double precision, but, in practice, the accumulated roundoff error using single precision is less than the error due to the polynomial approximation.

## (4) Incremental Rotation

- Let's back off from the approximation route and try another approach. Start with the vector (1,0) and multiply it by a one-degree rotation matrix each time throught the loop.

```python
DELTA = 2 * 3.14159 / 360.
SINA = SIN(DELTA)
COSA = COS(DELTA)
X = 1; Y = 0
MOVE(X, Y)
FOR I = 1 TO 360
    XNEW = X * COSA - Y * SINA
    Y = X * SINA + Y * COSA
    X = XNEW
    DRAW(X, Y)
```

## (5) Extrem Approximation

- If the incremental angle is small enough, we can approximate $$ cos\alpha = 1 $$ and $$ sin\alpha = a $$. The number of times through the loop is $$ n = 2 \pi / n $$, depending on which you want to use as input.

```python
A = .015; N = 2 * 3.14159 / A
X = 1; Y = 0
MOVE(X, Y)
FOR I = 1 TO N
    XNEW = X - Y * A
    Y = X * A + Y
    X = XNEW
    DRAW(X, Y)
```

- But there's a problem. Each time through the loop we are forming the product

$$ \left [ {x_{new}}, {y_{new}} \right ] = \left [ {x_{old}}, {y_{old}} \right ] 
\begin{bmatrix}
1 & a\\ 
-a & 1
\end{bmatrix} $$

- The matrix is almost a rotation matrix, but its determinant equals $$ 1 + a^2 $$. This is bad. It means that the running $$ \left [ x, y \right ] $$ can be magnified by this amount on each iteration, so what we get is a spiral that gets bigger and bigger. How to fix this? Introduce a bug into the algorithm.

## (6) Unskewing the Approximation

- Since vector multiplication and assignment don't occur in one statement, we had to calculate y carefully, using the old value for x. Suppose we were dumb and did it the naive way.

```python
A = .015; N = 2 * 3.14159 / A
X = 1; Y = 0
MOVE(X, Y)
FOR I = 1 TO N
X = X - Y * A
Y = X * A + Y
DRAW(X, Y)
```

- Now, what is the effect of this? Really what we get is

$$ x_{new} = x_{old} - y_{old} a $$

$$ y_{new} = x_{new} a + y_{old} = x_{old} a + y_{old} (1 - a^2) $$

- In other words,

$$ 
\left [ x_{new}, y_{new} \right ] = \left [ x_{old} , y_{old} \right ] 
\begin{bmatrix}
1 & a\\ 
-a & 1-a^2
\end{bmatrix} $$

- This matrix has a determinant of 1, and there is no net sprialing effect. What you get is actually an ellipse that is stretched slightly in the northest-southwest direction and squeezed slightly in the northwest-southeast direction. The maximum radius error in these directions is approximately a/4.

- Now comes the interesting part. Since you can start out with any vector, let's try(1000, 0). Now, cleverly select a to be an inverse power of 2 and the multiplication becomes just a shift. For example, a value of a = 1/64 is just a right shift by 6 bits. This generates the circle in about 402 steps. So, you can do this all with just integer arithmetic and no multiplication. This, children, is how we used to draw circles quickly - and in fact do rotation incrementally - before the age of hardware floating point and even hardware multiplication. (I understand that this was invented by Marvin Minsky.)

## (7) Rational Polynomials

- Another polynomial tack can be taken by looking in our hat and pulling out the following rabbit:

- &emsp; if &emsp; $$ x = (1-t^2) / (1+t^2) $$
- &emsp; and &emsp; $$ y = 2 t / (1 + t^2 ) $$
- &emsp; then &emsp; $$ x^2 + y^2 = 1 $$


- no matter what t is (or identically, as the mathematicians would say). Running t from 0 to 1 gives the upper-right quadrant of the circle. We can again evaluate these polynomials by forward differences, stepping t in increaments of .01, and get

```python
X = 1; DX = -.0001; DDX = -.0002
Y = 0; DY = .02
W = 1; DW = .0001; DDW = 0.0002
MOVE(X, Y)
FOR I = 1 TO 100
    X = X + DY; DX = DX + DDX
    Y = Y + DY
    W = W + DW; DW = DW + DDW
    DRAW(X/W, Y/W)
```

- Note that this is not an approximation like the last few tries. It is exact - except for roundoff error. Even roundoff error can be removed, either by calculating the polynomials directly or by scaling all numbers by 10000 and doing it with integers. (The division x/w must still be done in floating point.)

- This one has alwyas amazed me: you get to effectively evaluate two transcendental functions exactly with only a few additions. What's the catch? It's an application of the No-Free-Lunch Theorem-you don't get to pick the angles. If you watch the points, you see that they are not equally spaced around the circle. In fact, as t goes to indinity, the point keeps goint counterclockwise but slows down, finally running out of juice at (-1, 0). If you go backwards to minus infinity, the point goes clockwise, finally stopping again at(-1, 0). (Yet more evidence that $$-\infty  $$ to $$+\infty $$). To draw a complete circle, you are best advised to run t from -1 to +1, which draws the whole right half, and then mirror it to get the left half.

## (8) Differential Equation

- An entirely different technique is to describe the motion of [x, y] dynamically. Imaging the point rotating about the center as a function of time t. The position, velocity, and acceleration of the point will be


$$ \left [ x, y \right ] = \left [ cos t, sin t \right ] $$

