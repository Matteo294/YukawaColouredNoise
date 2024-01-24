cmake_minimum_required(VERSION 3.10)

project(AutoCorrelaton)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	#set(CMAKE_CXX_FLAGS "-O2 -ftree-vectorize -march=native -Wall -Wextra -fopenmp -pedantic -Wno-unknown-pragmas -Wno-unused-parameter")
	set(CMAKE_CXX_FLAGS "-O2 -ftree-vectorize -march=native -Wall -Wextra -Weffc++")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(CMAKE_CXX_FLAGS "-O2 -msse4 -ipo -xHost -flto -ftree-vectorize -Wall -Wextra -qopenmp")
endif()

include(CMakeParseArguments)
include(FindPackageHandleStandardArgs)

function(preferential_find_path VAR)
	cmake_parse_arguments(
		preferential_find_path
		""
		""
		"NAMES;PATHS;PATH_SUFFIXES"
		${ARGN}
	)

	string(REPLACE "~" "$ENV{HOME}" PATHS2 "${${preferential_find_path_PATHS}}")

	find_path(${VAR}
		NAMES ${preferential_find_path_NAMES}
		PATHS ${PATHS2}
		PATH_SUFFIXES ${preferential_find_path_PATH_SUFFIXES}
		NO_DEFAULT_PATH
	)
	if (${${VAR}} STREQUAL ${VAR}-NOTFOUND)
		find_path(${VAR}
			NAMES ${preferential_find_path_NAMES}
			PATH_SUFFIXES ${preferential_find_path_PATH_SUFFIXES}
			)
	endif()

	find_package_handle_standard_args(${preferential_find_path_NAMES}
		"Coult NOT find include directory for ${preferential_find_path_NAMES}. Please specify option -D${preferential_find_path_PATHS}=<path>."
		${VAR}
	)
endfunction(preferential_find_path)

function(preferential_find_library VAR)
	cmake_parse_arguments(
		preferential_find_library
		""
		""
		"NAMES;PATHS;PATH_SUFFIXES"
		${ARGN}
	)

	string(REPLACE "~" "$ENV{HOME}" PATHS2 "${${preferential_find_library_PATHS}}")

	find_library(${VAR}
		NAMES ${preferential_find_library_NAMES}
		PATHS ${PATHS2}
		PATH_SUFFIXES ${preferential_find_library_PATH_SUFFIXES}
		NO_DEFAULT_PATH
	)
	if (${${VAR}} STREQUAL ${VAR}-NOTFOUND)
		find_library(${VAR}
			NAMES ${preferential_find_library_NAMES}
			PATH_SUFFIXES ${preferential_find_library_PATH_SUFFIXES}
			)
	endif()
	find_package_handle_standard_args(${preferential_find_library_NAMES}
		"Coult NOT find library ${preferential_find_library_NAMES}. Please specify option -D${preferential_find_library_PATHS}=<path>."
		${VAR}
	)
endfunction(preferential_find_library)

preferential_find_path(FFTW_INCLUDE_DIRS
	NAMES fftw3.h
	PATHS FFTWPath
	PATH_SUFFIXES include api inc include/x86_64 include/x64
	)

preferential_find_library(FFTW_LIBRARIES
	NAMES libfftw3.so
	PATHS FFTWPath
	PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86
	)

preferential_find_library(MATHS_LIBRARIES
	NAMES m
	PATHS MathPath
	)

add_library(ac SHARED ac.cpp)
add_library(acStatic STATIC ac.cpp)

target_link_libraries(ac PUBLIC ${FFTW_LIBRARIES} ${MATHS_LIBRARIES})
target_link_libraries(acStatic PUBLIC ${FFTW_LIBRARIES} ${MATHS_LIBRARIES})

set_target_properties(acStatic PROPERTIES OUTPUT_NAME ac)
