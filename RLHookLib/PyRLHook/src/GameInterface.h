#pragma once
#define NOMINMAX 
#include <windows.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <map>
#include <atomic>
#include <thread>
#include <optional>
#include <gl/GL.h>
#include "src/Pipe.h"


namespace py = pybind11;

enum PixelFormat {
    RED = GL_RED,
    GREEN = GL_GREEN,
    BLUE = GL_BLUE,
    ALPHA = GL_ALPHA,
    RGB = GL_RGB,
    RGBA = GL_RGBA,
};

enum PixelDataType {
    UBYTE = GL_UNSIGNED_BYTE,
    FLOAT32 = GL_FLOAT
};

class GameInterface
{

public:
	explicit GameInterface(DWORD pid, PixelFormat pixelFormat, PixelDataType pixelDataType);
	explicit GameInterface(const std::string &processName, PixelFormat pixelFormat, PixelDataType pixelDataType);
	~GameInterface();

	std::optional<py::array> step(bool readPixelBuffer);
	std::uint32_t bufferSize() const;

    void setSpeed(std::double_t speed);
    void runAfap(std::double_t targetFramerate);

    void checkForException();

	template <typename T> T read(const std::string &mod, const std::vector<DWORD> &offsets);
	template <typename T> T read(LPCVOID adr);
	template <typename T> void write(const std::string &mod, const std::vector<DWORD> &offsets, const T &value);
	template <typename T> void write(LPVOID adr, const T &value);

	static std::string basePath;

private:

    void detachDll();
    void checkIfFinished() const;

    void finish();

	HANDLE gameProcess = INVALID_HANDLE_VALUE;
	DWORD pid = 0;
	Pipe pipe;
	Pipe timeDistorterPipe;

	std::map<std::string, LPVOID> moduleBases;

    HANDLE exceptionMailSlot = INVALID_HANDLE_VALUE;

    HANDLE hMapFile = INVALID_HANDLE_VALUE;
    LPVOID pBuf = nullptr;
	std::uint32_t width = 0;
	std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::uint32_t dataTypeSize = 0;
    PixelFormat pixelFormat;
    PixelDataType pixelDataType;

	bool isFinished = false;
};

template<typename T>
T GameInterface::read(const std::string &mod, const std::vector<DWORD> &offsets) {
    if (moduleBases.find(mod) == moduleBases.end())
        moduleBases[mod] = Utils::getModuleBaseAddress(gameProcess, mod);
    return Utils::template read<T>(gameProcess, moduleBases[mod], offsets);
}

template<typename T>
T GameInterface::read(LPCVOID adr) {
    return Utils::template read<T>(gameProcess, adr);
}

template<typename T>
void GameInterface::write(const std::string &mod, const std::vector<DWORD> &offsets, const T &value) {
    if (moduleBases.find(mod) == moduleBases.end())
        moduleBases[mod] = Utils::getModuleBaseAddress(gameProcess, mod);
    Utils::template write<T>(gameProcess, moduleBases[mod], offsets, value);
}

template<typename T>
void GameInterface::write(LPVOID adr, const T & value) {
    Utils::template write<T>(gameProcess, adr, value);
}
