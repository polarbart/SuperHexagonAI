
#include "GameInterfaceDll.h"
#include <string>
#include "TimeInterfaceDll.h"
#include "Utils.h"

FunSwapBuffers GameInterface::tWglSwapBuffers = nullptr;

GLenum GameInterface::glReadPixelFormat = 0;
GLenum GameInterface::glReadPixelType = 0;

std::uint32_t GameInterface::width = 0;
std::uint32_t GameInterface::height = 0;
std::uint32_t GameInterface::channels = 0;
std::uint32_t GameInterface::dataTypeSize = 0;

HANDLE GameInterface::hMapFile = INVALID_HANDLE_VALUE;
LPVOID GameInterface::pBuf = nullptr;

HANDLE GameInterface::exceptionMsFile = INVALID_HANDLE_VALUE;

std::atomic_bool GameInterface::isFinished = false;
HANDLE GameInterface::isOutOfRenderHook = INVALID_HANDLE_VALUE;

std::atomic_bool GameInterface::clientConnected = false;

Pipe GameInterface::pipe;
std::thread GameInterface::listenForClient;


void GameInterface::throwException(const char* message) {
    if (exceptionMsFile != INVALID_HANDLE_VALUE) {
        DWORD bytesWritten;
        WriteFile(exceptionMsFile, message, static_cast<DWORD>(strlen(message) * sizeof(char)), &bytesWritten, nullptr);
    }
    // std::cout << message << std::endl;
    onDetach();
}

BOOL GameInterface::initExceptionMsFile() {
    if (exceptionMsFile == INVALID_HANDLE_VALUE) {
        exceptionMsFile = CreateFile(EXCEPTION_MS_NAME, GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (exceptionMsFile == INVALID_HANDLE_VALUE) {
            DWORD err = GetLastError();
            std::string message = "Initializing the exception mailslot failed with the Windows error " + std::to_string(err) + ". ";
            if (err == ERROR_ACCESS_DENIED)
                message += "Does this process have admin rights? ";
            message += "DLL injection failed!";
            MessageBox(nullptr, message.c_str(), "GameInterfaceDll", MB_OK);
            return FALSE;
        }
    }
    return TRUE;
}

BOOL GameInterface::onAttach() {

    // init exception mailslot and hook to "initFromRenderThread"

    BOOL result = initExceptionMsFile();
    if (!result)
        return FALSE;

    try {

        // init ipc
        pipe = Pipe(PIPE_NAME, true);

        tWglSwapBuffers = reinterpret_cast<FunSwapBuffers>(DetourFindFunction("opengl32.dll", "wglSwapBuffers"));
        Utils::checkError(tWglSwapBuffers == NULL, "DetourFindFunction");

        Utils::checkDetoursError(DetourTransactionBegin(), "DetourTransactionBegin");
        Utils::checkDetoursError(DetourUpdateThread(GetCurrentThread()), "DetourUpdateThread");
        Utils::checkDetoursError(DetourAttach(&(PVOID &) tWglSwapBuffers, &(PVOID &) initFromRenderThread), "DetourAttach");
        Utils::checkDetoursError(DetourTransactionCommit(), "DetourTransactionCommit");

    } catch (const std::exception& e) {
        throwException(e.what());
        return FALSE;
    }
    return TRUE;
}

void GameInterface::initFromRenderThread(HDC arg) {
    try {

        // call hooked function
        tWglSwapBuffers(arg);

        // detach this function so it is only called once
        Utils::checkDetoursError(DetourTransactionBegin(), "DetourTransactionBegin");
        Utils::checkDetoursError(DetourUpdateThread(GetCurrentThread()), "DetourUpdateThread");
        Utils::checkDetoursError(DetourDetach(&(PVOID &) tWglSwapBuffers, &(PVOID &) initFromRenderThread), "DetourDetach");
        tWglSwapBuffers = nullptr;
        Utils::checkDetoursError(DetourTransactionCommit(), "DetourTransactionCommit");

        isOutOfRenderHook = CreateEvent(nullptr, TRUE, FALSE, nullptr);
        Utils::checkError(isOutOfRenderHook == NULL, "CreateEvent");

        // get width and height of screen
        // this has to be done from the render thread
        // and therefore can't be don in "communicateWHCBs"
        GLint m_viewport[4];
        glGetIntegerv(GL_VIEWPORT, m_viewport);
        Utils::checkOpenGLError("glGetIntegerv");
        width = m_viewport[2];
        height = m_viewport[3];

        // attach hook
        tWglSwapBuffers = reinterpret_cast<FunSwapBuffers>(DetourFindFunction("opengl32.dll", "wglSwapBuffers"));
        Utils::checkError(tWglSwapBuffers == nullptr, "DetourFindFunction");

        Utils::checkDetoursError(DetourTransactionBegin(), "DetourTransactionBegin");
        Utils::checkDetoursError(DetourUpdateThread(GetCurrentThread()), "DetourUpdateThread");
        Utils::checkDetoursError(DetourAttach(&(PVOID &) tWglSwapBuffers, &(PVOID &) hWglSwapBuffers), "DetourAttach");
        Utils::checkDetoursError(DetourTransactionCommit(), "DetourTransactionCommit");

        TimeInterface::listen();
        listenForGameInterfaceClient();

    } catch (const std::exception &e) {
        // redirect every exception to the python process
        throwException(e.what());
    }
}

void GameInterface::listenForGameInterfaceClient() {
    if (listenForClient.joinable())
        listenForClient.join();
    listenForClient = std::thread(connectToClient);
}

void GameInterface::connectToClient() {
    while (true) {
        try {
            // wait for some client to connect
            pipe.connect([] { return !isFinished; });

            // init exception mailslot if not initialized
            initExceptionMsFile();

            // communicate parameters
            communicateWHCBs();

            // create shared memory
            hMapFile = CreateFileMapping(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                         width * height * channels * dataTypeSize, SHARED_MEMORY_NAME);
            Utils::checkError(hMapFile == nullptr, "CreateFileMapping");
            pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, width * height * channels * dataTypeSize);
            Utils::checkError(pBuf == nullptr, "MapViewOfFile");

            // notify init finished
            pipe.write(true);

            clientConnected = true;

        } catch (const WindowsError &e) {
            if (e.errorCode == ERROR_BROKEN_PIPE)
                continue;
            if (!isFinished)
                throwException(e.what());
            break;
        } catch (const std::exception &e) {
            if (!isFinished)
                throwException(e.what());
            break;
        }
        break;
    }
}

void GameInterface::communicateWHCBs() {

    // receive parameter
    glReadPixelFormat = pipe.read<GLenum>();
    glReadPixelType = pipe.read<GLenum>();

    // compute number of channels
    switch (glReadPixelFormat) {
        case GL_RED:
        case GL_GREEN:
        case GL_BLUE:
        case GL_ALPHA:
            channels = 1;
            break;
        case GL_RGB:
            channels = 3;
            break;
        case GL_RGBA:
            channels = 4;
            break;
        default:
            throw std::runtime_error("Invalid pixel format");
    }

    // compute datatype size
    switch (glReadPixelType) {
        case GL_UNSIGNED_BYTE:
            dataTypeSize = 1;
            break;
        case GL_FLOAT:
            dataTypeSize = 4;
            break;
        default:
            throw std::runtime_error("Invalid pixel data type");
    }

    // send dimensions of screen
    pipe.write(width);
    pipe.write(height);
    pipe.write(channels);
    pipe.write(dataTypeSize);
}

void GameInterface::onDetach() {
    isFinished = true;

    onClientDisconnect();

    TimeInterface::detach();

    if (tWglSwapBuffers != nullptr) {
        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        DetourDetach(&(PVOID &) tWglSwapBuffers, &(PVOID &) hWglSwapBuffers);
        DetourTransactionCommit();
        tWglSwapBuffers = nullptr;
    }

    if (isOutOfRenderHook != INVALID_HANDLE_VALUE) {
        WaitForSingleObject(isOutOfRenderHook, 1000);
        // just to make sure that the game thread is out of this module, wait for another 100ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CloseHandle(isOutOfRenderHook);
        isOutOfRenderHook = INVALID_HANDLE_VALUE;
    }

    pipe = Pipe();

    if (listenForClient.joinable() && std::this_thread::get_id() != listenForClient.get_id())
        listenForClient.join();

}

void GameInterface::onClientDisconnect() {

    clientConnected = false;

    CloseHandle(hMapFile);
    hMapFile = INVALID_HANDLE_VALUE;

	UnmapViewOfFile(pBuf);
	pBuf = nullptr;

	CloseHandle(exceptionMsFile);
	exceptionMsFile = INVALID_HANDLE_VALUE;

}

void WINAPI GameInterface::hWglSwapBuffers(HDC arg) {
    static bool readPixelBuffer = true;
    DWORD ret;
    try {
        ret = ResetEvent(isOutOfRenderHook);
        Utils::checkError(ret == NULL, "ResetEvent");

        TimeInterface::advanceFrame();

        if (readPixelBuffer && clientConnected) {
            glReadBuffer(GL_BACK);
            Utils::checkOpenGLError("glReadBuffer");
            glReadPixels(0, 0, width, height, glReadPixelFormat, glReadPixelType, pBuf);
            Utils::checkOpenGLError("glReadPixels");
        }

        tWglSwapBuffers(arg);

        if (clientConnected) {
            pipe.write(true);
            readPixelBuffer = pipe.read<bool>([] { return !isFinished; });
        }

        ret = SetEvent(isOutOfRenderHook);
        Utils::checkError(ret == NULL, "SetEvent");

    } catch (const WindowsError &e) {
        if (e.errorCode == ERROR_BROKEN_PIPE) { // client disconnected
            onClientDisconnect();
            listenForGameInterfaceClient();
        } else if (!isFinished)
            throwException(e.what());
        SetEvent(isOutOfRenderHook);
    } catch (const std::exception &e) {
        if (!isFinished)
            throwException(e.what());
        SetEvent(isOutOfRenderHook);
    }
}
