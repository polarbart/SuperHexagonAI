
#ifndef RLHOOKLIB_GAMEINTERFACEDLL_H
#define RLHOOKLIB_GAMEINTERFACEDLL_H

#include <iostream>
#include <Windows.h>
#include <detours.h>
#include <thread>
#include <gl/GL.h>
#include "src/Pipe.h"

#define EXCEPTION_MS_NAME "\\\\.\\mailslot\\rlhook-exception"
#define SHARED_MEMORY_NAME "Global\\rlhook-sharedmemory"
#define PIPE_NAME "\\\\.\\pipe\\rlhook-pipe"

typedef void  (WINAPI *FunSwapBuffers)(HDC);

class GameInterface {
public:

    static std::atomic_bool isFinished;

    static BOOL onAttach();
    static void onDetach();

    static void throwException(const char* message);

private:
    static FunSwapBuffers tWglSwapBuffers;

    static GLenum glReadPixelFormat;
    static GLenum glReadPixelType;

    static std::uint32_t width;
    static std::uint32_t height;
    static std::uint32_t channels;
    static std::uint32_t dataTypeSize;

    static HANDLE hMapFile;
    static LPVOID pBuf;

    static HANDLE exceptionMsFile;

    static HANDLE isOutOfRenderHook;

    static std::atomic_bool clientConnected;

    static Pipe pipe;
    static std::thread listenForClient;

    static void WINAPI hWglSwapBuffers(HDC arg);
    static void WINAPI initFromRenderThread(HDC arg);

    static BOOL initExceptionMsFile();

    static void communicateWHCBs();
    static void onClientDisconnect();

    static void listenForGameInterfaceClient();
    static void connectToClient();
};

#endif //RLHOOKLIB_GAMEINTERFACEDLL_H
