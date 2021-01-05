
#ifndef RLHOOKLIB_EXCEPTIONS_H
#define RLHOOKLIB_EXCEPTIONS_H

#include <iostream>
#include <string>

struct Timeout : public std::exception {

public:
    const std::string whatMessage;

    Timeout(const std::string& functionName, std::uint32_t ms) : whatMessage(functionName + " timed out after " + std::to_string(ms) + "ms!") {}

    [[nodiscard]] const char* what() const override {
        return whatMessage.c_str();
    }
};

struct ErrorCodeException : public std::exception {

public:
    const std::string whatMessage;
    const DWORD errorCode;

    ErrorCodeException(std::string whatMessage, DWORD errorCode) : whatMessage(std::move(whatMessage)), errorCode(errorCode) {}

    [[nodiscard]] const char* what() const override {
        return whatMessage.c_str();
    }
};


struct WindowsError : public ErrorCodeException {

public:
    WindowsError(const std::string& functionName, DWORD errorCode) : ErrorCodeException(
            (errorCode == 5 ? "Does this process have admin rights? " : "") +
            functionName + " failed with the Windows error " + std::to_string(errorCode) + ".", errorCode) {}

};

struct DetoursError : public ErrorCodeException {

public:

    DetoursError(const std::string& functionName, DWORD errorCode)
    : ErrorCodeException(functionName + " failed with the Detours error " + std::to_string(errorCode) + ".", errorCode) {}

};

struct OpenGlError : public ErrorCodeException {

public:

    OpenGlError(const std::string& functionName, DWORD errorCode)
    : ErrorCodeException(functionName + " failed with the OpenGL error " + std::to_string(errorCode) + ".", errorCode) {}

};

struct ReturnError : public ErrorCodeException {

public:

    ReturnError(const std::string& functionName, DWORD errorCode)
    : ErrorCodeException(functionName + " did not exit gracefully (return code: " + std::to_string(errorCode) + ").", errorCode) {}

};

#endif //RLHOOKLIB_EXCEPTIONS_H
