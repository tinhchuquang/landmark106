#pragma once
#include <NvInfer.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#define CHECK(status) \
    if (status != 0) { \
        std::cerr << "Cuda failure: " << cudaGetErrorString(status) << std::endl; \
        abort(); \
    }

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) std::cout << msg << std::endl;
    }
};

extern Logger gLogger;

struct TRTModel {
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
};

std::vector<char> readEngineFile(const std::string& engineFile);

TRTModel loadModel(const std::string& engine_file, nvinfer1::ILogger& logger);
void destroyModel(TRTModel& model);
void printBindings(const TRTModel& model);