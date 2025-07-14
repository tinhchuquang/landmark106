#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>
#include <fstream>

#define CHECK(status) \
    if (status != 0) { \
        std::cerr << "Cuda failure: " << cudaGetErrorString(status) << std::endl; \
        abort(); \
    }

// Đọc engine từ file
inline std::vector<char> readEngineFile(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open engine file");
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

// Logger đơn giản
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) std::cout << msg << std::endl;
    }
};

extern Logger gLogger;