//
// Created by Julius on 01.12.2020.
//

#include <stdexcept>
#include <iostream>
#include "IsAlive.h"
#include "Exceptions.h"

#define ARE_YOU_ALIVE(x) ("Global\\" + (x) + "-areYouAlive").c_str()
#define I_AM_ALIVE(x) ("Global\\" + (x) + "-iAmAlive").c_str()

IsAlive::IsAlive(IsAlive &&other) noexcept : areYouAlive(other.areYouAlive), iAmAlive(other.iAmAlive) {
    other.areYouAlive = other.iAmAlive = INVALID_HANDLE_VALUE;
}

IsAlive &IsAlive::operator=(IsAlive &&other) noexcept {
    if (this == &other)
        return *this;

    CloseHandle(areYouAlive);
    CloseHandle(iAmAlive);

    areYouAlive = other.areYouAlive;
    iAmAlive = other.iAmAlive;

    other.areYouAlive = other.iAmAlive = INVALID_HANDLE_VALUE;

    return *this;
}

IsAlive::~IsAlive() {
    CloseHandle(areYouAlive);
    CloseHandle(iAmAlive);
}

IsAliveRequester::IsAliveRequester(const std::string &name) {
    areYouAlive = OpenEvent(EVENT_MODIFY_STATE, FALSE, ARE_YOU_ALIVE(name));
    if (areYouAlive == nullptr)
        throw WindowsError("OpenEvent", GetLastError());

    iAmAlive = OpenEvent(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, I_AM_ALIVE(name));
    if (iAmAlive == nullptr)
        throw WindowsError("OpenEvent", GetLastError());
}

bool IsAliveRequester::isAlive(DWORD timeout) {
    DWORD result = SignalObjectAndWait(areYouAlive, iAmAlive, timeout, FALSE);
    switch (result) {
        case WAIT_TIMEOUT:
            return false;
        case WAIT_OBJECT_0:
            return true;
        case WAIT_FAILED:
            throw WindowsError("SignalObjectAndWait", GetLastError());
        default:
            throw ReturnError("SignalObjectAndWait", result);
    }
}

IsAliveResponder::IsAliveResponder(const std::string &name) {
    areYouAlive = CreateEvent(nullptr, FALSE, FALSE, ARE_YOU_ALIVE(name));
    if (areYouAlive == nullptr)
        throw WindowsError("CreateEvent", GetLastError());

    iAmAlive = CreateEvent(nullptr, FALSE, FALSE, I_AM_ALIVE(name));
    if (iAmAlive == nullptr)
        throw WindowsError("CreateEvent", GetLastError());
}

IsAliveResponder::~IsAliveResponder() {
    stopResponder();
}

void IsAliveResponder::startResponder() {
    if (isResponding)
        return;
    isResponding = true;
    stopResponding = false;
    responderThread = std::thread([this]() {
        while (!stopResponding) {
            DWORD result = WaitForSingleObject(areYouAlive, 10);
            if (result == WAIT_OBJECT_0)
                SetEvent(iAmAlive);
        }
    });
}

void IsAliveResponder::stopResponder() {
    if (!stopResponding) {
        stopResponding = true;
        responderThread.join();
    }
}
