//
// Created by Julius on 29.11.2020.
//

#ifndef RLHOOKLIB_PIPE_H
#define RLHOOKLIB_PIPE_H

#include <Windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include "Exceptions.h"

typedef std::function<bool()> BoolFunction;

class Pipe {
public:

    Pipe() = default;
    Pipe(const Pipe&) = delete;
    Pipe(Pipe&& p) noexcept;
    Pipe& operator=(Pipe&& p) noexcept;

    Pipe(const std::string& name, bool isServer, DWORD waitNamedPipeTimeout=10000);
    ~Pipe();
    void connect(DWORD timeout=INFINITE);
    void connect(BoolFunction retryUntil);

    template <typename T>
    T read(DWORD timeout=INFINITE) {
        T data;
        readBytes(&data, sizeof(T), timeout);
        return data;
    }

    template <typename T>
    T read(BoolFunction retryUntil) {
        T data;
        readBytes(&data, sizeof(T), retryUntil);
        return data;
    }

    template <typename T>
    void readBytes(T* data, DWORD nBytes, DWORD timeout=INFINITE) {
        readBytes(data, nBytes, timeout, std::nullopt);
    }

    template <typename T>
    void readBytes(T* data, DWORD nBytes, BoolFunction retryUntil) {
        readBytes(data, nBytes, 10, std::optional(retryUntil));
    }

    template <typename T>
    void write(const T& data, DWORD timeout=INFINITE) {
        writeBytes(&data, sizeof(T), timeout);
    }

    template <typename T>
    void write(const T& data, BoolFunction retryUntil) {
        writeBytes(&data, sizeof(T), retryUntil);
    }

    template <typename T>
    void writeBytes(const T* data, DWORD nBytes, DWORD timeout=INFINITE) {
        writeBytes(data, nBytes, timeout, std::nullopt);
    }

    template <typename T>
    void writeBytes(const T* data, DWORD nBytes, const BoolFunction& retryUntil) {
        writeBytes(data, nBytes, 10, std::optional(retryUntil));
    }

    void finish();

private:
    HANDLE file = INVALID_HANDLE_VALUE;
    bool isServer = false;

    void funWithTimeout(const std::string& funName, const std::function<BOOL(OVERLAPPED&)>& fun, DWORD timeout, std::optional<BoolFunction> retryUntil);

    void connect(DWORD timeout, std::optional<BoolFunction> retryUntil);

    template <typename T>
    void writeBytes(const T* data, DWORD nBytes, DWORD timeout, const std::optional<BoolFunction>& retryUntil) {
        auto fun = [this, &data, &nBytes](OVERLAPPED& ol){return WriteFile(file, data, nBytes, nullptr, &ol);};
        funWithTimeout("WriteFile", fun, timeout, retryUntil);
    }

    template <typename T>
    void readBytes(T* data, DWORD nBytes, DWORD timeout, const std::optional<BoolFunction>& retryUntil) {
        auto fun = [this, &data, &nBytes](OVERLAPPED& ol){return ReadFile(file, data, nBytes, nullptr, &ol);};
        funWithTimeout("ReadFile", fun, timeout, retryUntil);
    }

};


#endif //RLHOOKLIB_PIPE_H
