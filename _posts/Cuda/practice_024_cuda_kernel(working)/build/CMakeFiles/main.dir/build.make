# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build"

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main_generated_main.cu.o: CMakeFiles/main.dir/main_generated_main.cu.o.depend
CMakeFiles/main.dir/main_generated_main.cu.o: CMakeFiles/main.dir/main_generated_main.cu.o.cmake
CMakeFiles/main.dir/main_generated_main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir="/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/main.dir/main_generated_main.cu.o"
	cd "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles/main.dir" && /usr/bin/cmake -E make_directory "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles/main.dir//."
	cd "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles/main.dir" && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D "generated_file:STRING=/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles/main.dir//./main_generated_main.cu.o" -D "generated_cubin_file:STRING=/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles/main.dir//./main_generated_main.cu.o.cubin.txt" -P "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles/main.dir//main_generated_main.cu.o.cmake"

# Object files for target main
main_OBJECTS =

# External object files for target main
main_EXTERNAL_OBJECTS = \
"/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles/main.dir/main_generated_main.cu.o"

main: CMakeFiles/main.dir/main_generated_main.cu.o
main: CMakeFiles/main.dir/build.make
main: /usr/lib/x86_64-linux-gnu/libcudart_static.a
main: /usr/lib/x86_64-linux-gnu/librt.so
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main

.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend: CMakeFiles/main.dir/main_generated_main.cu.o
	cd "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)" "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)" "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build" "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build" "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_024_cuda_kernel(working)/build/CMakeFiles/main.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/main.dir/depend

