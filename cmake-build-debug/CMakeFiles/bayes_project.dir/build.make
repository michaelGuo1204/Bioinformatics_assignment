# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = /home/bili/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/202.8194.17/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/bili/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/202.8194.17/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/bili/Document/Bioinformatics/Bayes_Project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/bayes_project.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bayes_project.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bayes_project.dir/flags.make

CMakeFiles/bayes_project.dir/main.c.o: CMakeFiles/bayes_project.dir/flags.make
CMakeFiles/bayes_project.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/bayes_project.dir/main.c.o"
	mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/bayes_project.dir/main.c.o   -c /media/bili/Document/Bioinformatics/Bayes_Project/main.c

CMakeFiles/bayes_project.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/bayes_project.dir/main.c.i"
	mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /media/bili/Document/Bioinformatics/Bayes_Project/main.c > CMakeFiles/bayes_project.dir/main.c.i

CMakeFiles/bayes_project.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/bayes_project.dir/main.c.s"
	mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /media/bili/Document/Bioinformatics/Bayes_Project/main.c -o CMakeFiles/bayes_project.dir/main.c.s

# Object files for target bayes_project
bayes_project_OBJECTS = \
"CMakeFiles/bayes_project.dir/main.c.o"

# External object files for target bayes_project
bayes_project_EXTERNAL_OBJECTS =

bayes_project: CMakeFiles/bayes_project.dir/main.c.o
bayes_project: CMakeFiles/bayes_project.dir/build.make
bayes_project: src/bin/libcpart.so
bayes_project: cuda/bin/libCuda.so
bayes_project: /usr/lib/x86_64-linux-gnu/libmpich.so
bayes_project: /usr/local/cuda-10.2/lib64/libcudart_static.a
bayes_project: /usr/lib/x86_64-linux-gnu/librt.so
bayes_project: CMakeFiles/bayes_project.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable bayes_project"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bayes_project.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bayes_project.dir/build: bayes_project

.PHONY : CMakeFiles/bayes_project.dir/build

CMakeFiles/bayes_project.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bayes_project.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bayes_project.dir/clean

CMakeFiles/bayes_project.dir/depend:
	cd /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/bili/Document/Bioinformatics/Bayes_Project /media/bili/Document/Bioinformatics/Bayes_Project /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/CMakeFiles/bayes_project.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bayes_project.dir/depend

