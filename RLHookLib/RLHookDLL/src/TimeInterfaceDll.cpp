
#include "TimeInterfaceDll.h"
#include "GameInterfaceDll.h"
#include "Utils.h"

std::thread TimeInterface::listenThread;

FunTimeGetTime TimeInterface::tTimeGetTime = nullptr;
FunGetTickCount TimeInterface::tGetTickCount = nullptr;
FunGetTickCount64 TimeInterface::tGetTickCount64 = nullptr;
FunQueryPerformanceCounter TimeInterface::tQueryPerformanceCounter = nullptr;

TimeDistorter<DWORD> TimeInterface::tdTimeGetTime;
TimeDistorter<DWORD> TimeInterface::tdGetTickCount;
TimeDistorter<ULONGLONG> TimeInterface::tdGetTickCount64;
TimeDistorter<LONGLONG> TimeInterface::tdQueryPerformanceCounter;


void TimeInterface::init() {

    tdTimeGetTime = TimeDistorter<DWORD>(1000);
    tdGetTickCount = TimeDistorter<DWORD>(1000);
    tdGetTickCount64 = TimeDistorter<ULONGLONG>(1000);
    LARGE_INTEGER li;
    QueryPerformanceFrequency(&li);
    tdQueryPerformanceCounter = TimeDistorter<LONGLONG>(static_cast<std::double_t>(li.QuadPart));

    tTimeGetTime = reinterpret_cast<FunTimeGetTime>(DetourFindFunction("Winmm.dll", "timeGetTime"));
    Utils::checkError(tTimeGetTime == NULL, "DetourFindFunction");
    tGetTickCount = reinterpret_cast<FunGetTickCount>(DetourFindFunction("kernel32.dll", "GetTickCount"));
    Utils::checkError(tGetTickCount == NULL, "DetourFindFunction");
    tGetTickCount64 = reinterpret_cast<FunGetTickCount64>(DetourFindFunction("kernel32.dll", "GetTickCount64"));
    Utils::checkError(tGetTickCount64 == NULL, "DetourFindFunction");
    tQueryPerformanceCounter = reinterpret_cast<FunQueryPerformanceCounter>(DetourFindFunction("ntdll.dll", "RtlQueryPerformanceCounter"));
    Utils::checkError(tQueryPerformanceCounter == NULL, "DetourFindFunction");

    Utils::checkDetoursError(DetourTransactionBegin(), "DetourTransactionBegin");
    Utils::checkDetoursError(DetourUpdateThread(GetCurrentThread()), "DetourUpdateThread");
    Utils::checkDetoursError(DetourAttach(&(PVOID&) tTimeGetTime, &(PVOID&) hTimeGetTime), "DetourAttach");
    Utils::checkDetoursError(DetourAttach(&(PVOID&) tGetTickCount, &(PVOID&) hGetTickCount), "DetourAttach");
    Utils::checkDetoursError(DetourAttach(&(PVOID&) tGetTickCount64, &(PVOID&) hGetTickCount64), "DetourAttach");
    Utils::checkDetoursError(DetourAttach(&(PVOID&) tQueryPerformanceCounter, &(PVOID&) hQueryPerformanceCounter), "DetourAttach");
    Utils::checkDetoursError(DetourTransactionCommit(), "DetourTransactionCommit");
}

void TimeInterface::detach() {
    if (tTimeGetTime != nullptr || tGetTickCount != nullptr || tGetTickCount64 != nullptr || tQueryPerformanceCounter != nullptr) {
        DetourTransactionBegin();
        DetourUpdateThread(GetCurrentThread());
        if (tTimeGetTime != nullptr)
            DetourDetach(&(PVOID &) tTimeGetTime, &(PVOID &) hTimeGetTime);
        if (tGetTickCount != nullptr)
            DetourDetach(&(PVOID &) tGetTickCount, &(PVOID &) hGetTickCount);
        if (tGetTickCount64 != nullptr)
            DetourDetach(&(PVOID &) tGetTickCount64, &(PVOID &) hGetTickCount64);
        if (tQueryPerformanceCounter != nullptr)
            DetourDetach(&(PVOID &) tQueryPerformanceCounter, &(PVOID &) hQueryPerformanceCounter);
        DetourTransactionCommit();
    }
    tTimeGetTime = nullptr;
    tGetTickCount = nullptr;
    tGetTickCount64 = nullptr;
    tQueryPerformanceCounter = nullptr;

    if (listenThread.joinable() && std::this_thread::get_id() != listenThread.get_id())
        listenThread.join();

}
void TimeInterface::listen() {

    init();

    listenThread = std::thread([]{ // disconnect

        Pipe pipe(TIME_DISTORTER_PIPE_NAME, true);

        while (!GameInterface::isFinished) {
            try {
                pipe.connect([]() { return !GameInterface::isFinished; });

                while (!GameInterface::isFinished) {
                    std::uint8_t function = pipe.read<std::uint8_t>([]() { return !GameInterface::isFinished; });
                    switch (function) {
                        case 0:
                            setSpeed(pipe.read<std::double_t>());
                            break;
                        case 1:
                            setTargetFrameRate(pipe.read<std::double_t>());
                            break;
                        default:
                            break;
                    }
                }

            } catch (const WindowsError &e) {
                if (!GameInterface::isFinished && e.errorCode != ERROR_BROKEN_PIPE) {
                    GameInterface::throwException(e.what());
                    break;
                }
            } catch (const std::exception &e) {
                if (!GameInterface::isFinished) {
                    GameInterface::throwException(e.what());
                    break;
                }
            }
            if (!GameInterface::isFinished)
                setSpeed(1);
        }
    });
}

void TimeInterface::advanceFrame() {

    /*
    static int frames = 0;
    static DWORD lastTime = tTimeGetTime();
    DWORD diff = tTimeGetTime() - lastTime;
    ++frames;
    if (diff > 1000) {
        std::cout << "FPS: " << ((float)frames) / diff * 1000 << std::endl;
        frames = 0;
        lastTime = tTimeGetTime();
    }
     */

    tdTimeGetTime.advanceFrame();
    tdGetTickCount.advanceFrame();
    tdGetTickCount64.advanceFrame();
    tdQueryPerformanceCounter.advanceFrame();
}

void TimeInterface::setSpeed(std::double_t s) {
    tdTimeGetTime.setSpeed(s, tTimeGetTime());
    tdGetTickCount.setSpeed(s, tGetTickCount());
    tdGetTickCount64.setSpeed(s, tGetTickCount64());
    LARGE_INTEGER li;
    tQueryPerformanceCounter(&li);
    tdQueryPerformanceCounter.setSpeed(s, li.QuadPart);
}

void TimeInterface::setTargetFrameRate(std::double_t framerate) {
    tdTimeGetTime.setTargetFramerate(framerate, tTimeGetTime());
    tdGetTickCount.setTargetFramerate(framerate, tGetTickCount());
    tdGetTickCount64.setTargetFramerate(framerate, tGetTickCount64());
    LARGE_INTEGER li;
    tQueryPerformanceCounter(&li);
    tdQueryPerformanceCounter.setTargetFramerate(framerate, li.QuadPart);
}


DWORD WINAPI TimeInterface::hTimeGetTime() {
    return tdTimeGetTime.distort(tTimeGetTime());
}

DWORD WINAPI TimeInterface::hGetTickCount() {
    return tdGetTickCount.distort(tGetTickCount());
}

ULONGLONG WINAPI TimeInterface::hGetTickCount64() {
    return tdGetTickCount64.distort(tGetTickCount64());
}

BOOL WINAPI TimeInterface::hQueryPerformanceCounter(LARGE_INTEGER *lpPerformanceCount) {
    BOOL result = tQueryPerformanceCounter(lpPerformanceCount);
    lpPerformanceCount->QuadPart = tdQueryPerformanceCounter.distort(lpPerformanceCount->QuadPart);
    return result;
}
