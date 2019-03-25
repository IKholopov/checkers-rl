cmake_minimum_required(VERSION 3.0)

project(Checkers-ai)
file(GLOB CheckersAI_SRC
    "*.h"
    "*.cpp"
)

file(GLOB CheckersAI_SWIG_SRC
    "*.i"
)

set(CheckersAI_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Checkers-ai include directories")
set(CMAKE_CXX_STANDARD 17)

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
find_package(SWIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(${SWIG_USE_FILE})

add_library( checkers_ai ${CheckersAI_SRC} )
set_property(TARGET checkers_ai PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(checkers_ai PUBLIC ${CheckersAI_DIR})

SET(CMAKE_SWIG_FLAGS "")

set_property(SOURCE ${CheckersAI_SWIG_SRC} PROPERTY CPLUSPLUS ON USE_LIBRARY_INCLUDE_DIRECTORIES TRUE
                                                    COMPILE_OPTIONS py3)
swig_add_library( checkers_swig TYPE SHARED LANGUAGE python SOURCES ${CheckersAI_SWIG_SRC})
target_include_directories( checkers_swig PUBLIC ${Python_INCLUDE_DIRS} ${CheckersAI_DIR} )
target_link_libraries(checkers_swig PUBLIC checkers_ai)
set_property(TARGET checkers_swig PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
set_property(TARGET checkers_swig PROPERTY SWIG_GENERATED_INCLUDE_DIRECTORIES ${Python_INCLUDE_DIRS})
set_property(TARGET checkers_swig PROPERTY SWIG_INCLUDE_DIRECTORIES ${Python_INCLUDE_DIRS})
set_property(TARGET checkers_swig PROPERTY GENERATED_INCLUDE_DIRECTORIES ${Python_INCLUDE_DIRS})

file(GLOB CheckersAI_Py
    "*.py"
)
foreach(py_src ${CheckersAI_Py})
    get_filename_component(py_name ${py_src} NAME)
    configure_file(${py_src} ${CMAKE_CURRENT_BINARY_DIR}/${shader_name})
endforeach(py_src)
