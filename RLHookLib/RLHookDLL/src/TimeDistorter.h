//
// Created by Julius on 05.06.2020.
//

#ifndef RLHOOKLIB_TIMEDISTORTER_H
#define RLHOOKLIB_TIMEDISTORTER_H

#include <cmath>
#include <iostream>

template <typename T>
class TimeDistorter {
public:
    TimeDistorter() = default;
    explicit TimeDistorter(std::double_t timeUnitsPerSecond);

    void setSpeed(std::double_t s, T time);
    void setTargetFramerate(std::double_t frameRate, T time);
    T distort(T time);
    void advanceFrame();

private:

    std::double_t timeUnitsPerSecond = 0;

    T baseTime = 0;

    T lastTimeSetSpeed = 0;
    std::double_t speed = 1;

    std::double_t timePerFrame = 0;
    std::uint64_t nFrames = 0;

    enum Mode {NONE, SET_SPEED, TARGET_FRAMERATE};
    Mode mode = NONE;
};

template<typename T>
TimeDistorter<T>::TimeDistorter(std::double_t timeUnitsPerSecond) : timeUnitsPerSecond(timeUnitsPerSecond) {}

template<typename T>
void TimeDistorter<T>::setSpeed(std::double_t s, T time) {
    if (mode == NONE && s == 1)
        return;
    baseTime = distort(time);
    mode = SET_SPEED;
    lastTimeSetSpeed = time;
    speed = s;
}

template<typename T>
void TimeDistorter<T>::setTargetFramerate(std::double_t frameRate, T time) {
    baseTime = distort(time);
    mode = TARGET_FRAMERATE;
    timePerFrame = timeUnitsPerSecond / frameRate;
    nFrames = 1;
}

template<typename T>
T TimeDistorter<T>::distort(T time) {
    switch (mode) {
        case NONE:
            return time;
        case SET_SPEED:
            return baseTime + (T) (speed * (time - lastTimeSetSpeed));
        case TARGET_FRAMERATE:
            return baseTime + (T) (timePerFrame * nFrames);
        default:
            throw std::runtime_error("Invalid mode");
    }
}

template<typename T>
void TimeDistorter<T>::advanceFrame() {
    nFrames++;
}

#endif //RLHOOKLIB_TIMEDISTORTER_H
