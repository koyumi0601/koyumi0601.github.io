---
layout: single
title: "How to add equation on a post"
categories: blog
tags: [Blog, Markdown]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

# Work Instruction

- Add file _includes/mathjax_support.html


```html
<script type="text/x-mathjax-config">
	MathJax.Hub.Config({
	TeX: {
		equationNumbers: {
		autoNumber: "AMS"
		}
	},
	tex2jax: {
		inlineMath: [ ['$', '$'] ],
		displayMath: [ ['$$', '$$'] ],
		processEscapes: true,
		}
	});
	MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
		alert("Math Processing Error: "+message[1]);
	});
	MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
		alert("Math Processing Error: "+message[1]);
	});
</script>

<script type="text/javascript" async
	src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```



- Add lines in file _include/scripts.html

```
<script type="text/javascript" async
	src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML">
</script>

<script type="text/x-mathjax-config">
	MathJax.Hub.Config({
	extensions: ["tex2jax.js"],
	jax: ["input/TeX", "output/HTML-CSS"],
	tex2jax: {
		inlineMath: [ ['$','$'], ["\\(","\\)"] ],
		displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
		processEscapes: true
	},
	"HTML-CSS": { availableFonts: ["TeX"] }
});
</script>
```



- Edit file _layouts/default.html

![2023-08-04_22-35-math-layout]({{site.url}}/images/2023-08-03-Github-Blog-Posting-Equation/2023-08-04_22-35-math-layout.png)



- Conversion site: [https://latex.codecogs.com/eqneditor/editor.php](https://latex.codecogs.com/eqneditor/editor.php)

![2023-08-04_15-18-eq]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-04_15-18-eq.png)

- Wrap equation with &&  && like below

![2023-08-04_15-29-equation3]({{site.url}}/images/2023-08-02-Github-Blog-Posting/2023-08-04_15-29-equation3.png)
- Result

$$
\begin{matrix}
a & b & c \\
d & e & f
\end{matrix}
$$


# Reference

- [https://an-seunghwan.github.io/github.io/mathjax-error/](https://an-seunghwan.github.io/github.io/mathjax-error/)
- [https://ashki23.github.io/markdown-latex.html](https://ashki23.github.io/markdown-latex.html)