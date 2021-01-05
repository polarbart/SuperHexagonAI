set(PY_RL_HOOK "pyrlhook")

pybind11_add_module(${PY_RL_HOOK} src/PythonBindings.cpp src/GameInterface.cpp src/Utils.cpp)
target_link_libraries(${PY_RL_HOOK} PRIVATE pybind11::embed OpenGL::GL Utils)
target_include_directories(${PY_RL_HOOK} PRIVATE ../Utils)

