//
// Created by Julius on 29.11.2020.
//

#include "Pipe.h"

#include <utility>

Pipe::Pipe(const std::string &name, bool isServer, DWORD waitNamedPipeTimeout) : isServer(isServer) {
    if (isServer) {

        file = CreateNamedPipe(
                name.c_str(),
                PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
                1,
                1024 * 16,
                1024 * 16,
                NMPWAIT_USE_DEFAULT_WAIT,
                nullptr
        );

        if (file == INVALID_HANDLE_VALUE)
            throw WindowsError("CreateNamedPipe", GetLastError());
    } else {

        while (true) {
            file = CreateFile(
                    name.c_str(),
                    GENERIC_READ | GENERIC_WRITE,
                    0,
                    nullptr,
                    OPEN_EXISTING,
                    FILE_FLAG_OVERLAPPED,
                    nullptr
            );

            if (file == INVALID_HANDLE_VALUE) {
                DWORD err = GetLastError();
                if (err == ERROR_PIPE_BUSY) {
                    DWORD res = WaitNamedPipe(name.c_str(), waitNamedPipeTimeout);
                    if (res != NULL)
                        continue;
                    throw Timeout("WaitNamedPipe", waitNamedPipeTimeout);
                }
                throw WindowsError("CreateFile", err);
            }
            break;
        }
    }
}

Pipe::Pipe(Pipe &&p) noexcept : file(p.file), isServer(p.isServer) {
    p.file = INVALID_HANDLE_VALUE;
}

Pipe& Pipe::operator=(Pipe &&p) noexcept {
    if (this == &p)
        return *this;
    finish();
    file = p.file;
    isServer = p.isServer;
    p.file = INVALID_HANDLE_VALUE;
    return *this;
}

Pipe::~Pipe() {
    finish();
}

void Pipe::connect(DWORD timeout) {
    connect(timeout, std::nullopt);
}

void Pipe::connect(BoolFunction retryUntil) {
    connect(10, std::optional(retryUntil));
}

void Pipe::connect(DWORD timeout, std::optional<BoolFunction> retryUntil) {
    auto fun = [this](OVERLAPPED& ol){return ConnectNamedPipe(file, &ol);};
    funWithTimeout("ConnectNamedPipe", fun, timeout, std::move(retryUntil));
}

void Pipe::finish() {
    CancelIo(file);
    CloseHandle(file);
    file = INVALID_HANDLE_VALUE;
}

void Pipe::funWithTimeout(const std::string &funName, const std::function<BOOL(OVERLAPPED &)>& fun, DWORD timeout, std::optional<BoolFunction> retryUntil) {
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Pipe is already destroyed.");

    OVERLAPPED ol = {0};

    BOOL ret = fun(ol);
    if (!ret) {
        DWORD err = GetLastError();
        try {
            switch (err) {
                case ERROR_SUCCESS:
                case ERROR_PIPE_CONNECTED:
                    break;
                case ERROR_IO_PENDING:
                    while (true) {
                        DWORD numberOfBytesTransferred = 0;
                        DWORD ret2 = GetOverlappedResultEx(file, &ol, &numberOfBytesTransferred, timeout, FALSE);
                        if (!ret2) {
                            DWORD err2 = GetLastError();
                            if (err2 == WAIT_TIMEOUT) {
                                if (retryUntil.has_value() && retryUntil.value()())
                                    continue;
                                throw Timeout(funName, timeout);
                            } else
                                throw WindowsError("GetOverlappedResultEx", err2);
                        }
                        break;
                    }
                    break;
                default:
                    throw WindowsError(funName, err);
            }
        } catch (const WindowsError &e) {
            if (e.errorCode == ERROR_BROKEN_PIPE && isServer) {
                DisconnectNamedPipe(file);
                std::cout << "closing handle " << file << std::endl;
            }
            throw;
        }
    }
}
