---
layout: single
title: "How to make batchfile, open vscode with specific folder"
categories: blog
tags: [Blog, Markdown]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true
---

*배치파일 만들기*



@echo off
start "" "C:\Users\YourUsername\AppData\Local\Programs\Microsoft VS Code\Code.exe" "D:\YourFolderPath"
exit