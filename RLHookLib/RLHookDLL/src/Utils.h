
#ifndef RLHOOKLIB_UTILS_H
#define RLHOOKLIB_UTILS_H

class Utils {
public:
    static void checkError(bool error, const std::string& funName);
    static void checkDetoursError(LONG ret, const std::string& funName);
    static void checkOpenGLError(const std::string& funName);
};

#endif //RLHOOKLIB_UTILS_H
