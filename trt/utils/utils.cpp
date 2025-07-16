#include "utils.h"

Logger gLogger;

std::vector<char> readEngineFile(const std::string& engineFile) {
    std::ifstream file(engineFile, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Failed to open engine file");
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

TRTModel loadModel(const std::string& engine_file, nvinfer1::ILogger& logger) {
    TRTModel model;
    std::vector<char> engine_data = readEngineFile(engine_file);
    model.runtime = nvinfer1::createInferRuntime(logger);
    model.engine = model.runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    model.context = model.engine->createExecutionContext();
    return model;
}

void destroyModel(TRTModel& model) {
    if (model.context) model.context->destroy();
    if (model.engine) model.engine->destroy();
    if (model.runtime) model.runtime->destroy();
}

void printBindings(const TRTModel& model) {
    auto* engine = model.engine;
    int nbBindings = engine->getNbBindings();
    std::cout << "=== TensorRT Bindings ===" << std::endl;
    for (int i = 0; i < nbBindings; ++i) {
        const char* name = engine->getBindingName(i);
        bool is_input = engine->bindingIsInput(i);
        std::cout << (is_input ? "[Input] " : "[Output] ") << "Binding " << i << ": " << name << std::endl;

        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        std::cout << "  Shape: [";
        for (int j = 0; j < dims.nbDims; ++j) {
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "=========================" << std::endl;
}