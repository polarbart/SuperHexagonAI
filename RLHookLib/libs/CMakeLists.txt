add_subdirectory(Detours)
if (NOT DEFINED ONLY_ADDITIONAL_BINARIES)
    add_subdirectory(pybind11)
endif()