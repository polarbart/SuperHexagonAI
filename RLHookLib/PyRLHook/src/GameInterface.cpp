
#include "GameInterface.h"
#include <chrono>
#include <filesystem>
#include "Utils.h"

#define OPEN_PROCESS_ACCESS_RIGHTS (PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ)

#define RL_HOOK_DLL_NAME std::string("RLHookDLL.dll")
#define RL_HOOK_DLL_PATH_x86 ("\\bin\\Win32\\" + RL_HOOK_DLL_NAME)
#define RL_HOOK_DLL_PATH_x64 ("\\bin\\x64\\" + RL_HOOK_DLL_NAME)

#define TIME_DISTORTER_DLL_NAME std::string("TimeDistorter.dll")

#define EXCEPTION_MS_NAME "\\\\.\\mailslot\\rlhook-exception"
#define SHARED_MEMORY_NAME "Global\\rlhook-sharedmemory"
#define PIPE_NAME "\\\\.\\pipe\\rlhook-pipe"
#define TIME_DISTORTER_PIPE_NAME "\\\\.\\pipe\\rlhook-isalivepipe"

std::string GameInterface::basePath;

GameInterface::GameInterface(
        const std::string &processName,
        PixelFormat pixelFormat,
        PixelDataType pixelDataType
) : GameInterface(Utils::getProcessPid(processName), pixelFormat, pixelDataType) {}

GameInterface::GameInterface(DWORD pid,
        PixelFormat pixelFormat,
        PixelDataType pixelDataType
) : pid(pid), pixelFormat(pixelFormat), pixelDataType(pixelDataType) {

    try {

        // get handle to target process
        gameProcess = OpenProcess(OPEN_PROCESS_ACCESS_RIGHTS, FALSE, pid);
        Utils::checkError(gameProcess == nullptr, "OpenProcess");

        // check if the game uses OpenGL
        HMODULE openGlModule = Utils::getModuleBaseAddress(gameProcess, "Opengl32.dll");
        if (openGlModule == nullptr)
            throw std::runtime_error("Does the game use OpenGL? Could not find \"Opengl32.dll\" within the target process.");

        // create mailslot to which the target process can communicate exceptions
        exceptionMailSlot = Utils::createMailslot(EXCEPTION_MS_NAME);

        // inject dll into target and attach pipe
        bool isx86 = Utils::isWow64Process(gameProcess);
        std::string dllPath = basePath + (isx86 ? RL_HOOK_DLL_PATH_x86 : RL_HOOK_DLL_PATH_x64);

        if (!std::filesystem::exists(dllPath))
            throw std::runtime_error(RL_HOOK_DLL_NAME + " wasn't found at \"" + dllPath + "\".");

        HMODULE dllModule = Utils::getModuleBaseAddress(gameProcess, RL_HOOK_DLL_NAME);
        if (dllModule == nullptr) {
            Utils::attachDll(gameProcess, dllPath);
            pipe = Pipe(PIPE_NAME, false);
        } else {
            try {
                pipe = Pipe(PIPE_NAME, false);
            } catch (const WindowsError& e) {
                if (e.errorCode != ERROR_FILE_NOT_FOUND)
                    throw;
                detachDll();
                Utils::attachDll(gameProcess, dllPath);
                pipe = Pipe(PIPE_NAME, false);
            }
        }

        // write parameters
        pipe.write<GLenum>(pixelFormat);
        pipe.write<GLenum>(pixelDataType);

        // receive parameters
        width = pipe.read<std::uint32_t>();
        height = pipe.read<std::uint32_t>();
        channels = pipe.read<std::uint32_t>();
        dataTypeSize = pipe.read<std::uint32_t>();

        // wait for init in target process to finish
        pipe.read<bool>();

        // open shared memory
        hMapFile = OpenFileMapping(
                FILE_MAP_READ,
                FALSE,
                SHARED_MEMORY_NAME
        );
        Utils::checkError(hMapFile == nullptr, "OpenFileMapping");

        pBuf = MapViewOfFile(
                hMapFile,
                FILE_MAP_READ,
                0,
                0,
                bufferSize()
        );
        Utils::checkError(pBuf == nullptr, "MapViewOfFile");

        // wait for glSwapBuffers to be called for the first time
        pipe.read<bool>();

        // attach pipe to time distorter
        timeDistorterPipe = Pipe(TIME_DISTORTER_PIPE_NAME, false);
    } catch (...) {
        if (exceptionMailSlot != INVALID_HANDLE_VALUE)
            checkForException();
        throw;
    }
}

GameInterface::~GameInterface() {
    finish();
}

void GameInterface::finish() {
    isFinished = true;

    pipe = Pipe();
    timeDistorterPipe = Pipe();

    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    CloseHandle(gameProcess);
    CloseHandle(exceptionMailSlot);

    pBuf = nullptr;
    hMapFile = INVALID_HANDLE_VALUE;
    gameProcess = INVALID_HANDLE_VALUE;
    exceptionMailSlot = INVALID_HANDLE_VALUE;
}

void GameInterface::detachDll() {
    HMODULE dllModule = Utils::getModuleBaseAddress(gameProcess, RL_HOOK_DLL_NAME);
    if (dllModule != nullptr)
        Utils::detachDll(gameProcess, dllModule);
}

void GameInterface::checkIfFinished() const {
    if (isFinished)
        throw std::runtime_error("This object was already destructed, probably due to an exception!");
}

std::optional<py::array> GameInterface::step(bool readPixelBuffer) {
    checkIfFinished();

    checkForException();

    try {
        // request the next frame
        pipe.write(readPixelBuffer);
        // wait for the game to advance one fame
        pipe.read<bool>();
    } catch (...) {
        checkForException();
        throw;
    }

    if (!readPixelBuffer)
        return std::nullopt;

    // return the screen which was written into the shared memory 'pBuf'
    return py::array(
            pixelDataType == UBYTE ? py::dtype::of<std::uint8_t>() : py::dtype::of<std::float_t>(),
            {height, width, channels},
            {width * channels * dataTypeSize, channels * dataTypeSize, dataTypeSize},
            pBuf
    );
}

void GameInterface::setSpeed(std::double_t speed) {
    checkIfFinished();
    checkForException();
    try {
        timeDistorterPipe.write<std::uint8_t>(0);
        timeDistorterPipe.write(speed);
    } catch (...) {
        checkForException();
        throw;
    }
}

void GameInterface::runAfap(std::double_t targetFramerate) {
    checkIfFinished();
    checkForException();
    try {
        timeDistorterPipe.write<std::uint8_t>(1);
        timeDistorterPipe.write(targetFramerate);
    } catch (...) {
        checkForException();
    }
}

void GameInterface::checkForException() {
    DWORD cbMessage, cbRead;
    BOOL result = GetMailslotInfo(exceptionMailSlot, nullptr, &cbMessage, nullptr, nullptr);
    try {
        Utils::checkError(!result, "GetMailslotInfo");
    } catch (...) {
        finish();
        throw;
    }
    if (cbMessage != MAILSLOT_NO_MESSAGE) {
        std::string message(cbMessage, '\0');
        result = ReadFile(exceptionMailSlot, &message[0], cbMessage, &cbRead, nullptr);

        try {
            Utils::checkError(!result, "ReadFile");
        } catch (...) {
            finish();
            throw;
        }
        finish();

        if (cbRead != cbMessage)
            throw std::runtime_error("Couldn't read exception mailslot");

        throw std::runtime_error("Remote process failed at some point with the following error message: \"" + message + "\"");
    }
}

std::uint32_t GameInterface::bufferSize() const {
    return width * height * channels * dataTypeSize;
}
