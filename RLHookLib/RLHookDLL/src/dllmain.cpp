
#include <windows.h>
#include <iostream>
#include "GameInterfaceDll.h"


BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
        case DLL_PROCESS_ATTACH:
            DisableThreadLibraryCalls(hModule);
            // AllocConsole();
            // freopen("conout$", "w", stdout);
            // std::cout << "DllMain#Attach" << std::endl;
            // std::cout << ul_reason_for_call << std::endl;
            return GameInterface::onAttach();
        case DLL_PROCESS_DETACH:
            // std::cout << ul_reason_for_call << std::endl;
            // std::cout << "DllMain#Detach" << std::endl;
            GameInterface::onDetach();
            // FreeConsole();
            break;
        default:
            // std::cout << ul_reason_for_call << std::endl;
            break;
    }
    return TRUE;
}

