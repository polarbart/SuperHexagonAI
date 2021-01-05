//
// Created by Julius on 01.12.2020.
//

#ifndef RLHOOKLIB_ISALIVE_H
#define RLHOOKLIB_ISALIVE_H

#include <Windows.h>
#include <string>
#include <thread>
#include <atomic>

class IsAlive {
public:
    IsAlive() = default;
    IsAlive(const IsAlive&) = delete;
    IsAlive(IsAlive&& other) noexcept;
    IsAlive& operator=(IsAlive&& other) noexcept;
    ~IsAlive();
protected:
    HANDLE iAmAlive = INVALID_HANDLE_VALUE;
    HANDLE areYouAlive = INVALID_HANDLE_VALUE;
};

class IsAliveRequester : IsAlive {
public:
    IsAliveRequester() = default;
    explicit IsAliveRequester(const std::string& name);
    bool isAlive(DWORD timeout = 100);
};

class IsAliveResponder : IsAlive {
public:
    IsAliveResponder() = default;
    explicit IsAliveResponder(const std::string& name);
    IsAliveResponder(const IsAliveResponder&) = delete;
    ~IsAliveResponder();
    void startResponder();
    void stopResponder();

private:
    std::thread responderThread;
    std::atomic<bool> stopResponding = true;
    bool isResponding = false;
};


#endif //RLHOOKLIB_ISALIVE_H
