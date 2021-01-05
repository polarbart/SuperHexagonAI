
#include <windows.h>
#include <string>
#include <gl/GL.h>
#include "src/Exceptions.h"
#include "Utils.h"

void Utils::checkError(bool error, const std::string& funName) {
    if (error)
        throw WindowsError(funName, GetLastError());
}

void Utils::checkDetoursError(LONG ret, const std::string& funName) {
    if (ret != NO_ERROR)
        throw DetoursError(funName, ret);
}

void Utils::checkOpenGLError(const std::string& funName) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
        throw OpenGlError(funName, err);
}
