#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include "utils.h"
#include "scrfd.h"


int getSize(const Dims& dims) {
    int s = 1;
    for (int i = 0; i < dims.nbDims; ++i) s *= dims.d[i];
    return s;
}

void preprocess_pfld(const cv::Mat& input_img, std::vector<float>& output_data){
    cv::Mat img_float;
    input_img.convertTo(img_float, CV_32FC3, 1.0 / 255);

    cv::Mat img_rgb;
    cv::cvtColor(img_float, img_rgb, cv::COLOR_BGR2RGB);

    int height = img_rgb.rows;
    int width = img_rgb.cols;

    output_data.resize(3 * height * width);

    // Split từng channel rồi đưa vào vector theo thứ tự CHW
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c)
        channels[c] = cv::Mat(height, width, CV_32F, output_data.data() + c * height * width);

    cv::split(img_rgb, channels); // đưa dữ liệu vào output_data dạng [C,H,W]
}


int main() {
    std::string engineFile = "../checkpoints/pfld/pfld.trt";
    std::string folder_path = "../test";

    TRTModel model = loadModel(engine_file, gLogger);
    // printBindings(model);
    int inputW = 640, inputH = 640;
    
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();

            if (file_path.ends_with(".jpg") || file_path.ends_with(".png")) {
                std::cout << "Đang đọc: " << file_path << std::endl;

                cv::Mat img = cv::imread(file_path);
                if (img.empty()) {
                    std::cerr << "❌ Không đọc được ảnh: " << file_path << std::endl;
                    continue;
                }
                auto faces = predictSCRFD(img, model, inputW, inputH);
                for (size_t i = 0; i < faces.size(), ++i){
                    const auto& face = faces[i];
                    int x1 = face.box.x;
                    int x2 = face.box.y;
                    int x2 = x1 + face.box.width - 1;
                    int y2 = y1 + face.box.height - 1;
                    int w = x2 - x1 + 1;
                    int h = y2 - y1 + 1;
                    int cx = x1 + w / 2;
                    int cy = y1 + h / 2;

                    int size = static_cast<int>(std::max(w, h) * 1.1);
                    x1 = cx - size / 2;
                    y1 = cy - size / 2;
                    x2 = x1 + size;
                    y2 = y1 + size;

                    int width = img.cols;
                    int height = img.rows;

                    // Tính phần vượt biên
                    int edx1 = std::max(0, -x1);
                    int edy1 = std::max(0, -y1);
                    int edx2 = std::max(0, x2 - width);
                    int edy2 = std::max(0, y2 - height);

                    // Clamp vào vùng ảnh gốc
                    int crop_x1 = std::max(0, x1);
                    int crop_y1 = std::max(0, y1);
                    int crop_x2 = std::min(width, x2);
                    int crop_y2 = std::min(height, y2);

                    cv::Mat cropped = img(cv::Rect(crop_x1, crop_y1, crop_x2 - crop_x1, crop_y2 - crop_y1)).clone();

                    // Padding nếu vượt biên
                    if (edx1 > 0 || edy1 > 0 || edx2 > 0 || edy2 > 0) {
                        cv::copyMakeBorder(cropped, cropped, edy1, edy2, edx1, edx2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                    }

                    // Resize về 112x112
                    cv::Mat input_resized;
                    cv::resize(cropped, input_resized, cv::Size(112, 112));

                    std::vector<float> chw_data;
                    preprocess_pfld(cropped);
                    

                }

            }
        }
    }

    int batchSize = 4;

    // Load engine
    auto engineData = readEngineFile(engineFile);
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Binding index
    int inputIndex = engine->getBindingIndex("input");
    int outputIndex = engine->getBindingIndex("output");
    int landmarksIndex = engine->getBindingIndex("landmarks");

    // Set dynamic batch shape
    context->setBindingDimensions(inputIndex, Dims4(batchSize, 3, 112, 112));
    assert(context->allInputDimensionsSpecified());

    // Get output shapes
    Dims outDims = context->getBindingDimensions(outputIndex);        // Bx64x28x28
    Dims lmDims = context->getBindingDimensions(landmarksIndex);     // Bx196

    size_t inputSize = batchSize * 3 * 112 * 112;
    size_t outputSize = batchSize * 64 * 28 * 28;
    size_t landmarksSize = batchSize * 196;

    // Allocate device buffers
    void* buffers[3];
    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float));
    cudaMalloc(&buffers[landmarksIndex], landmarksSize * sizeof(float));

    // (Optional) Copy input data to buffers[inputIndex] here

    // Run inference
    context->enqueueV2(buffers, 0, nullptr);

    // Copy output back to host (for example)
    std::vector<float> hostOutput(outputSize);
    std::vector<float> hostLandmarks(landmarksSize);
    cudaMemcpy(hostOutput.data(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostLandmarks.data(), buffers[landmarksIndex], landmarksSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few results
    std::cout << "First 5 landmark values: ";
    for (int i = 0; i < 5; ++i) std::cout << hostLandmarks[i] << " ";
    std::cout << std::endl;

    // Cleanup
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaFree(buffers[landmarksIndex]);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
