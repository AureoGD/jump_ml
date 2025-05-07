cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

#============================================================================
# Initialize the project
#============================================================================
project(rgc_controller)

#============================================================================
# Configure the project
#============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#============================================================================
# Find packages
#============================================================================
find_package(OsqpEigen REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(yaml-cpp REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(${PROJECT_NAME}
    SHARED
    src/model_matrices.cpp
    src/opt_problem.cpp
    src/pred_control.cpp
    src/opt_problem0.cpp
    src/opt_problem1.cpp
    src/opt_problem2.cpp
    # src/opt_problem3.cpp
    # src/opt_problem4.cpp
    # src/opt_problem5.cpp
    # src/opt_problem6.cpp
    # src/opt_problem7.cpp
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        OsqpEigen::OsqpEigen
        yaml-cpp
)

configure_file(${CMAKE_SOURCE_DIR}/config/config.yaml ${CMAKE_BINARY_DIR}/config/config.yaml COPYONLY)

set(BINDINGS_MODULE_NAME "pybind_opWrapper")

# Only source (.cpp) files should be listed here
pybind11_add_module(${BINDINGS_MODULE_NAME} MODULE
    "src/pybind_opWrapper.cpp"
	"src/op_wrapper.cpp"
)

target_link_libraries(${BINDINGS_MODULE_NAME}
    PRIVATE
        OsqpEigen::OsqpEigen
		${PROJECT_NAME}
)