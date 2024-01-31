---
layout: single
title: "Linux Terminal Command"
categories: setup
tags: [Ubuntu, Linux, Terminal]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*Terminal command, frequently used*



# Terminal Command

Unix (Lunux)

- Basic

```bash
man # search manual
clear # clear window
pwd # print working directory
ls # list folder and files in current directory
open . # open current directory in gui
cd # change directory
cd .. # upper directory
cd ~ # home directory
cd - # previous directory
find . -type file -name "*.txt" # find *.txt files in current directory
find . -type file -name "*.json" # find *.json files in current directory
find . -type directory -name "*2" # find * directory named *2 in current directory
which # find location of installed application
touch new_file1.txt # generate file
chmod 777 new_file1.txt # change permission read, write and execution for all user
cat new_file1.txt # print 1st line
echo "hello world" > new_file3.txt # print "hello world to new_file3.txt" (overwrite)
echo "goodbye world" >> new_file3.txt # append
mkdir dir3 # make directory
mkdir -p dir4/subdir1
```

- chmode 권한 777 755 가 무슨 뜻?

| 권한 숫자 | 권한         | 소유자(Owner) | 그룹(Group) | 다른 사용자(Other) |
|------------|--------------|---------------|-------------|--------------------|
| 7          | 읽기, 쓰기, 실행 | O             | O           | O                  |
| 6          | 읽기, 쓰기     | O             | O           |                    |
| 5          | 읽기, 실행     | O             | O           |                    |
| 4          | 읽기           | O             | O           |                    |
| 3          | 쓰기, 실행     | O             |             | O                  |
| 2          | 쓰기           | O             |             |                    |
| 1          | 실행           | O             |             |                    |
| 0          | 권한 없음     |               |             |                    |



- File

```bash
cp new_files1.txt dir3 # copy file
mv new_files3.txt dir4/ # move file
rm -r dir3 # remove directory
rm new_files1.txt # remove file
grep "world" *.txt # Global Regular Expression Print
grep -n "world" *.txt # print nth line including "world"
grep -ni "world" *.txt # grep insensitivite
grep -nir "world" .txt # grep insensitive and recursive
for dir in */; do mv "$dir" "Pop_$dir"; done # 폴더 이름 앞에 prefix 일괄 붙이기
```

- Environment variables

```bash
export MY_DIR="dir1" # set environment variable
env # print all environment variable
cd $MY_DIR # how to use; $MY_DIR
unset $MY_DIR # unset
```

- super user permission
```bash
sudo su
# password
```

- super user permission 2
```bash
sudo usermod -aG docker $USER
sudo reboot
```




# Reference

- terminal command [https://www.youtube.com/watch?v=EL6AQl-e3AQ](https://www.youtube.com/watch?v=EL6AQl-e3AQ)

