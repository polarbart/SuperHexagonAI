#include "Utils.h"
#include "GameInterface.h"

#include <string>
#include <tlhelp32.h>
#include <Shlwapi.h>
#include <sstream>
#include <psapi.h>


#define MODULE_NAME_BUFFER_SIZE 512
#define LOAD_LIBRARY ("LoadLibraryA")
#define FREE_LIBRARY ("FreeLibrary")
#define FUNCTION_ADDRESS_GETTER(x) (std::string("\\bin\\Win32\\FunctionAddressGetter.exe KERNEL32 ") + (x))

void Utils::checkError(bool error, const std::string &funName) {
    if (error) {
        DWORD errorCode = GetLastError();
        std::string message = funName + " failed with windows error " + std::to_string(errorCode);
        if (errorCode == 5)
            message = "Does this process have admin rights? " + message;
        throw std::runtime_error(message);
    }
}


// http://www.online-tutorials.net/security/dll-injection-injizieren-von-dlls-in-eine-oder-mehrere-zielanwendungen/tutorials-t-27-314.html
void Utils::attachDll(HANDLE targetProcess, const std::string &dllPath) {

    if (Utils::isWow64Process(GetCurrentProcess()) && !Utils::isWow64Process(targetProcess))
        throw std::runtime_error("Can't access 64 bit game from a 32 bit python process");

    // get the full path name of the dll and its length
    char fullDllPath[MAX_PATH + 1];
	DWORD fullDllPathSize = GetFullPathName(dllPath.c_str(), MAX_PATH + 1, fullDllPath, nullptr);
	Utils::checkError(fullDllPathSize == 0, "GetFullPathName");

    fullDllPathSize++;  // terminating null character
    // google python module additional resources
    // https://python-packaging.readthedocs.io/en/latest/non-code-files.html

    // allocate memory in the target process
	LPVOID alloc = VirtualAllocEx(targetProcess, nullptr, fullDllPathSize, MEM_COMMIT, PAGE_EXECUTE_READWRITE);

	// write the full path name into the memory of the target process
	BOOL success = WriteProcessMemory(targetProcess, alloc, fullDllPath, fullDllPathSize, nullptr);
	if (!success) {
	    DWORD lastError = GetLastError();
		VirtualFreeEx(targetProcess, alloc, fullDllPathSize, MEM_RELEASE);
        throw std::runtime_error("WriteProcessMemory failed with windows error " + std::to_string(lastError));
	}

	// execute the LoadLibrary function within the target process in order to load the dll
	LPTHREAD_START_ROUTINE addressLoadLibrary = Utils::getFunctionAddress(targetProcess, LOAD_LIBRARY);
	HANDLE remoteThread = CreateRemoteThread(targetProcess, nullptr, 0, addressLoadLibrary, alloc, 0, nullptr);
	if (remoteThread == nullptr) {
        DWORD lastError = GetLastError();
        VirtualFreeEx(targetProcess, alloc, fullDllPathSize, MEM_RELEASE);
        throw std::runtime_error("CreateRemoteThread failed with windows error " + std::to_string(lastError));
	}

    // get return value of the LoadLibrary function
    WaitForSingleObject(remoteThread, INFINITE);

    DWORD successLoadLibrary = 0;
	success = GetExitCodeThread(remoteThread, &successLoadLibrary);
    DWORD lastError = GetLastError();

	VirtualFreeEx(targetProcess, alloc, fullDllPathSize, MEM_RELEASE);
    CloseHandle(remoteThread);

    if (!success)
        throw std::runtime_error("GetExitCodeThread failed with windows error " + std::to_string(lastError));
    if (!successLoadLibrary)
        throw std::runtime_error("Failed to load dll into the target process.");
}

void Utils::detachDll(HANDLE targetProcess, const std::string &dllName) {
    HMODULE moduleHandle = getModuleBaseAddress(targetProcess, dllName);
    if (moduleHandle == nullptr)
        throw std::runtime_error("Couldn't find " + dllName + " within target process");
    detachDll(targetProcess, moduleHandle);
}

void Utils::detachDll(HANDLE targetProcess, HMODULE moduleHandle) {

    // execute the FreeLibrary function within the target process in order to detach the dll
    LPTHREAD_START_ROUTINE addressFreeLibrary = Utils::getFunctionAddress(targetProcess, FREE_LIBRARY);
    HANDLE remoteThread = CreateRemoteThread(targetProcess, nullptr, 0, addressFreeLibrary, moduleHandle, 0, nullptr);
    Utils::checkError(remoteThread == nullptr, "CreateRemoteThread");

    // get return value of the FreeLibrary function
    WaitForSingleObject(remoteThread, INFINITE);

    DWORD successFreeLibrary;
    DWORD success = GetExitCodeThread(remoteThread, &successFreeLibrary);
    DWORD lastError = GetLastError();

    CloseHandle(remoteThread);

    if (!success)
        throw std::runtime_error("GetExitCodeThread failed with windows error " + std::to_string(lastError));
    if (!successFreeLibrary)
        throw std::runtime_error("Failed to detach dll from the target process. Unfortunately the error code can't be retrieved.");
}

DWORD Utils::getProcessPid(const std::string &processName) {

	HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
	PROCESSENTRY32 structprocsnapshot = { 0 };

	structprocsnapshot.dwSize = sizeof(PROCESSENTRY32);

	if (snapshot == INVALID_HANDLE_VALUE)
		throw std::runtime_error("CreateToolhelp32Snapshot failed");

	if (!Process32First(snapshot, &structprocsnapshot))
		throw std::runtime_error("Couldn't find process with the specified name");
	do {
		if (!strcmp(structprocsnapshot.szExeFile, processName.c_str())) {
			CloseHandle(snapshot);
			return structprocsnapshot.th32ProcessID;
		}
	} while (Process32Next(snapshot, &structprocsnapshot));

	CloseHandle(snapshot);
	throw std::runtime_error("Couldn't find process with the specified name");
}

HMODULE Utils::getModuleBaseAddress(HANDLE targetProcess, const std::string &moduleName) {
    // get the number of modules within the target process
    std::vector<HMODULE> modules;
    DWORD cbNeeded1, cbNeeded2;
    do {
        BOOL success = EnumProcessModulesEx(targetProcess, nullptr, 0, &cbNeeded1, LIST_MODULES_ALL);
        Utils::checkError(!success, "EnumProcessModulesEx");

        std::int32_t numModules = cbNeeded1 / sizeof(HMODULE);
        modules = std::vector<HMODULE>(numModules);

        success = EnumProcessModulesEx(targetProcess, &modules[0], cbNeeded1, &cbNeeded2, LIST_MODULES_ALL);
        Utils::checkError(!success, "EnumProcessModulesEx");
    } while (cbNeeded1 != cbNeeded2);

    char moduleName2[MODULE_NAME_BUFFER_SIZE];
    if (moduleName.size() > MODULE_NAME_BUFFER_SIZE)
        throw std::runtime_error("getModuleBaseAddress failed since the module name was to long");

    for (auto &m : modules) {
        DWORD success = GetModuleBaseName(targetProcess, m, moduleName2, MODULE_NAME_BUFFER_SIZE);
        Utils::checkError(!success, "GetModuleBaseName");
        if (moduleName == moduleName2)
            return m;
    }
    return nullptr;
}

LPTHREAD_START_ROUTINE Utils::getFunctionAddress(HANDLE hProc, const std::string &funName) {
    if (Utils::isWow64Process(hProc) && !Utils::isWow64Process(GetCurrentProcess())) {
        DWORD result = std::system((GameInterface::basePath + FUNCTION_ADDRESS_GETTER(funName)).c_str());
        if (result < 0x00400000)
            throw std::runtime_error("FunctionAddressGetter failed");
        return (LPTHREAD_START_ROUTINE) result;
    }
    return (LPTHREAD_START_ROUTINE) GetProcAddress(GetModuleHandle("KERNEL32"), funName.c_str());
}

bool Utils::isWow64Process(HANDLE hProc) {
    BOOL result = false;
    BOOL success = IsWow64Process(hProc, &result);
    Utils::checkError(!success, "IsWow64Process");
    return result;
}

LPVOID Utils::resolveAddress(HANDLE hProc, LPCVOID base, const std::vector<DWORD> &offsets) {
    auto* tmp = (std::int8_t*) base;
    const std::uint8_t targetPointerSize = Utils::isWow64Process(hProc) ? 4 : 8;
    for (std::size_t i = 0; i < offsets.size() - 1; i++)
        ReadProcessMemory(hProc, tmp + offsets[i], &tmp, targetPointerSize, nullptr);
    return tmp;
}

HANDLE Utils::createMailslotFile(LPCSTR name) {
    HANDLE file = CreateFile(name, GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    Utils::checkError(file == INVALID_HANDLE_VALUE, "CreateFile");
    return file;
}

HANDLE Utils::createMailslot(LPCSTR name) {
    HANDLE mailslot = CreateMailslot(name, 0, MAILSLOT_WAIT_FOREVER, nullptr);
    Utils::checkError(mailslot == INVALID_HANDLE_VALUE, "CreateMailSlot");
    return mailslot;
}

HANDLE Utils::createEvent(LPCSTR name, BOOL initialState) {
    HANDLE event = CreateEvent(nullptr, FALSE, initialState, name);
    Utils::checkError(event == nullptr, "CreateEvent");
    return event;
}

void Utils::setEvent(HANDLE event) {
    DWORD result = SetEvent(event);
    Utils::checkError(!result, "SetEvent");
}
