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
include src/bin/CMakeFiles/mcmc.dir/depend.make

# Include the progress variables for this target.
include src/bin/CMakeFiles/mcmc.dir/progress.make

# Include the compile flags for this target's objects.
include src/bin/CMakeFiles/mcmc.dir/flags.make

src/bin/CMakeFiles/mcmc.dir/mcmc.c.o: src/bin/CMakeFiles/mcmc.dir/flags.make
src/bin/CMakeFiles/mcmc.dir/mcmc.c.o: ../src/mcmc.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object src/bin/CMakeFiles/mcmc.dir/mcmc.c.o"
	cd /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/src/bin && mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mcmc.dir/mcmc.c.o   -c /media/bili/Document/Bioinformatics/Bayes_Project/src/mcmc.c

src/bin/CMakeFiles/mcmc.dir/mcmc.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mcmc.dir/mcmc.c.i"
	cd /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/src/bin && mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /media/bili/Document/Bioinformatics/Bayes_Project/src/mcmc.c > CMakeFiles/mcmc.dir/mcmc.c.i

src/bin/CMakeFiles/mcmc.dir/mcmc.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mcmc.dir/mcmc.c.s"
	cd /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/src/bin && mpicc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /media/bili/Document/Bioinformatics/Bayes_Project/src/mcmc.c -o CMakeFiles/mcmc.dir/mcmc.c.s

# Object files for target mcmc
mcmc_OBJECTS = \
"CMakeFiles/mcmc.dir/mcmc.c.o"

# External object files for target mcmc
mcmc_EXTERNAL_OBJECTS =

src/bin/libmcmc.so: src/bin/CMakeFiles/mcmc.dir/mcmc.c.o
src/bin/libmcmc.so: src/bin/CMakeFiles/mcmc.dir/build.make
src/bin/libmcmc.so: src/bin/CMakeFiles/mcmc.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C shared library libmcmc.so"
	cd /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/src/bin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mcmc.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/bin/CMakeFiles/mcmc.dir/build: src/bin/libmcmc.so

.PHONY : src/bin/CMakeFiles/mcmc.dir/build

src/bin/CMakeFiles/mcmc.dir/clean:
	cd /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/src/bin && $(CMAKE_COMMAND) -P CMakeFiles/mcmc.dir/cmake_clean.cmake
.PHONY : src/bin/CMakeFiles/mcmc.dir/clean

src/bin/CMakeFiles/mcmc.dir/depend:
	cd /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/bili/Document/Bioinformatics/Bayes_Project /media/bili/Document/Bioinformatics/Bayes_Project/src /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/src/bin /media/bili/Document/Bioinformatics/Bayes_Project/cmake-build-debug/src/bin/CMakeFiles/mcmc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/bin/CMakeFiles/mcmc.dir/depend
