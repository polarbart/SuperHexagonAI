
#ifndef RLHOOKLIB_TIMEINTERFACEDLL_H
#define RLHOOKLIB_TIMEINTERFACEDLL_H

#include <iostream>

#include <Windows.h>
#include <detours.h>
#include <thread>
#include "src/Pipe.h"
#include "TimeDistorter.h"

#define TIME_DISTORTER_PIPE_NAME "\\\\.\\pipe\\rlhook-isalivepipe"

typedef DWORD (WINAPI *FunTimeGetTime)();
typedef DWORD (WINAPI *FunGetTickCount)();
typedef ULONGLONG (WINAPI *FunGetTickCount64)();
typedef BOOL (WINAPI *FunQueryPerformanceCounter)(LARGE_INTEGER*);

class TimeInterface {

public:

    static void listen();
    static void advanceFrame();

    static void detach();

private:

    static void init();

    static DWORD WINAPI hTimeGetTime();
    static DWORD WINAPI hGetTickCount();
    static ULONGLONG WINAPI hGetTickCount64();
    static BOOL WINAPI hQueryPerformanceCounter(LARGE_INTEGER *lpPerformanceCount);

    static void setSpeed(std::double_t s);
    static void setTargetFrameRate(std::double_t framerate);

    static std::thread listenThread;

    static FunTimeGetTime tTimeGetTime;
    static FunGetTickCount tGetTickCount;
    static FunGetTickCount64 tGetTickCount64;
    static FunQueryPerformanceCounter tQueryPerformanceCounter;

    static TimeDistorter<DWORD> tdTimeGetTime;
    static TimeDistorter<DWORD> tdGetTickCount;
    static TimeDistorter<ULONGLONG> tdGetTickCount64;
    static TimeDistorter<LONGLONG> tdQueryPerformanceCounter;

};

#endif //RLHOOKLIB_TIMEINTERFACEDLL_H
