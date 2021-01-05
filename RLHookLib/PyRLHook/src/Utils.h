#pragma once
#include <iostream>
#include <Windows.h>
#include <vector>


class Utils
{
public:

    static void checkError(bool error, const std::string &funName);

    template <typename T>
    static void writeMailslot(HANDLE file, const T &data);
    template <typename T>
    static bool readMailslot(HANDLE mailslot, T &message);
    static HANDLE createMailslotFile(LPCSTR name);
    static HANDLE createMailslot(LPCSTR name);

    static HANDLE createEvent(LPCSTR name, BOOL initialState);

    static void setEvent(HANDLE event);

	static void attachDll(HANDLE targetProcess, const std::string &dllPath);
	static void detachDll(HANDLE targetProcess, const std::string &dllName);
    static void detachDll(HANDLE targetProcess, HMODULE moduleHandle);
    static DWORD getProcessPid(const std::string &processName);
    static HMODULE getModuleBaseAddress(HANDLE targetProcess, const std::string &moduleName);

    static LPTHREAD_START_ROUTINE getFunctionAddress(HANDLE hProc, const std::string &funName);

    static bool isWow64Process(HANDLE hProc);

    static LPVOID resolveAddress(HANDLE hProc, LPCVOID base, const std::vector<DWORD> &offsets);

    template <typename T>
	static T read(HANDLE hProc, LPCVOID base, const std::vector<DWORD> &offsets);
	template <typename T>
	static T read(HANDLE hProc, LPCVOID adr);
	template <typename T>
	static void write(HANDLE hProc, LPVOID base, const std::vector<DWORD> &offsets, const T &value);
	template <typename T>
	static void write(HANDLE hProc, LPVOID adr, const T &value);
};

template<typename T>
void Utils::writeMailslot(HANDLE file, const T &data) {
    BOOL result;
    DWORD bytesWritten;
    result = WriteFile(file, &data, sizeof(T), &bytesWritten, nullptr);
    checkError(!result || bytesWritten != sizeof(T), "WriteFile");
}

template<typename T>
bool Utils::readMailslot(HANDLE mailslot, T &message) {

    DWORD cbMessage, cbRead;
    BOOL result = GetMailslotInfo(mailslot, nullptr, &cbMessage, nullptr, nullptr);

    Utils::checkError(!result, "GetMailslotInfo");

    if (cbMessage == MAILSLOT_NO_MESSAGE)
        return false;

    if (cbMessage != sizeof(T))
        throw std::runtime_error("Mailslot message has invalid size");

    result = ReadFile(mailslot, message, cbMessage, &cbRead, NULL);
    Utils::checkError(!result, "ReadFile");

    if (cbRead != sizeof(T))
        throw std::runtime_error("Couldn't read mailslot");

    return true;
}

template<typename T>
inline T Utils::read(HANDLE hProc, LPCVOID base, const std::vector<DWORD> &offsets) {
	return read<T>(hProc, (std::int8_t*) resolveAddress(hProc, base, offsets) + offsets[offsets.size() - 1]);
}

template<typename T>
inline T Utils::read(HANDLE hProc, LPCVOID adr) {
	T ret;
	ReadProcessMemory(hProc, adr, &ret, sizeof(T), NULL);
	return ret;
}

template<typename T>
inline void Utils::write(HANDLE hProc, LPVOID base, const std::vector<DWORD> &offsets, const T &value) { // mit richtigen pointern und handle übergeben
    write<T>(hProc, (std::int8_t*) resolveAddress(hProc, base, offsets) + offsets[offsets.size() - 1], value);
}

template<typename T>
inline void Utils::write(HANDLE hProc, LPVOID adr, const T &value) {
	WriteProcessMemory(hProc, adr, &value, sizeof(T), NULL);
}
