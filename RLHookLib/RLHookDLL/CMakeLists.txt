
set(RL_HOOK_DLL "RLHookDLL")

add_library(${RL_HOOK_DLL} SHARED src/dllmain.cpp src/GameInterfaceDll.cpp src/TimeDistorter.h src/TimeInterfaceDll.cpp src/Utils.cpp)
target_link_libraries(${RL_HOOK_DLL} PRIVATE Detours OpenGL::GL Utils)
target_include_directories(${RL_HOOK_DLL} PRIVATE ../Utils)
set_target_properties(${RL_HOOK_DLL} PROPERTIES LINK_FLAGS /EXPORT:DetourFinishHelperProcess,@1,NONAME)
