# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tsmc/teamip/lyz/PUCRN/evaluation_code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tsmc/teamip/lyz/PUCRN/evaluation_code

# Include any dependencies generated for this target.
include CMakeFiles/evaluation_lyz.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/evaluation_lyz.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/evaluation_lyz.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/evaluation_lyz.dir/flags.make

CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o: CMakeFiles/evaluation_lyz.dir/flags.make
CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o: evaluation_lyz.cpp
CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o: CMakeFiles/evaluation_lyz.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tsmc/teamip/lyz/PUCRN/evaluation_code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o -MF CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o.d -o CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o -c /home/tsmc/teamip/lyz/PUCRN/evaluation_code/evaluation_lyz.cpp

CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tsmc/teamip/lyz/PUCRN/evaluation_code/evaluation_lyz.cpp > CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.i

CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tsmc/teamip/lyz/PUCRN/evaluation_code/evaluation_lyz.cpp -o CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.s

# Object files for target evaluation_lyz
evaluation_lyz_OBJECTS = \
"CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o"

# External object files for target evaluation_lyz
evaluation_lyz_EXTERNAL_OBJECTS =

evaluation_lyz: CMakeFiles/evaluation_lyz.dir/evaluation_lyz.cpp.o
evaluation_lyz: CMakeFiles/evaluation_lyz.dir/build.make
evaluation_lyz: /usr/local/lib/libmpfr.so
evaluation_lyz: /usr/local/lib/libgmp.so
evaluation_lyz: /usr/local/lib/libCGAL.so.13.0.3
evaluation_lyz: /usr/local/lib/libmpfr.so
evaluation_lyz: /usr/local/lib/libgmp.so
evaluation_lyz: CMakeFiles/evaluation_lyz.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tsmc/teamip/lyz/PUCRN/evaluation_code/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable evaluation_lyz"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/evaluation_lyz.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/evaluation_lyz.dir/build: evaluation_lyz
.PHONY : CMakeFiles/evaluation_lyz.dir/build

CMakeFiles/evaluation_lyz.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/evaluation_lyz.dir/cmake_clean.cmake
.PHONY : CMakeFiles/evaluation_lyz.dir/clean

CMakeFiles/evaluation_lyz.dir/depend:
	cd /home/tsmc/teamip/lyz/PUCRN/evaluation_code && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tsmc/teamip/lyz/PUCRN/evaluation_code /home/tsmc/teamip/lyz/PUCRN/evaluation_code /home/tsmc/teamip/lyz/PUCRN/evaluation_code /home/tsmc/teamip/lyz/PUCRN/evaluation_code /home/tsmc/teamip/lyz/PUCRN/evaluation_code/CMakeFiles/evaluation_lyz.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/evaluation_lyz.dir/depend
