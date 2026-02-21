cmake_minimum_required(VERSION 3.16)
project(bg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----- CLI -----
add_executable(bg_cli
  src/main_cli.cpp
  src/env.cpp
  src/obs.cpp
  src/ascii.cpp
)
target_include_directories(bg_cli PRIVATE src)

if (MSVC)
  target_compile_options(bg_cli PRIVATE /W4 /O2)
else()
  target_compile_options(bg_cli PRIVATE -Wall -Wextra -Wpedantic -O2)
endif()

# ----- Python module (pybind11) -----
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
# pybind11 из pip: лучше искать через python -m pybind11 --includes не будем,
# вместо этого подключим include dir напрямую:
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE PYBIND11_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(bg_env MODULE
  src/bindings.cpp
  src/env.cpp
  src/obs.cpp
)
target_include_directories(bg_env PRIVATE src "${PYBIND11_INCLUDE}")
target_link_libraries(bg_env PRIVATE Python::Module)

# имя модуля без lib и с правильным суффиксом pyd
set_target_properties(bg_env PROPERTIES
  PREFIX ""
  SUFFIX ".pyd"
)

if (MSVC)
  target_compile_options(bg_env PRIVATE /W4 /O2)
else()
  target_compile_options(bg_env PRIVATE -Wall -Wextra -Wpedantic -O2)
endif()

# Куда складывать .pyd (рядом с python-скриптами)
set(PY_OUT_DIR "${CMAKE_SOURCE_DIR}/../python")
set_target_properties(bg_env PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PY_OUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${PY_OUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${PY_OUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${PY_OUT_DIR}"
)