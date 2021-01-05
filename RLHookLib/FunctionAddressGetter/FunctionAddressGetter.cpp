#include <iostream>
#include <Windows.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Usage: FunctionAddressGetter.exe [Module] [Function]" << std::endl;
        return 1;
    }
    return reinterpret_cast<int>(GetProcAddress(GetModuleHandle(argv[1]), argv[2]));
}
