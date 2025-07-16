#include <iostream>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "utils.h"
#include "scrfd_postprocess.h"
#include <algorithm>
#include <memory>
#include <cmath>

using namespace nvinfer1;

cv::Mat preprocess_img(const cv::Mat& img, cv::Size input_size, float& det_scale, int& new_width, int& new_height) {
    float im_ratio = float(img.rows) / img.cols;
    float model_ratio = float(input_size.height) / input_size.width;
    if (im_ratio > model_ratio) {
        new_height = input_size.height;
        new_width = int(new_height / im_ratio);
    } else {
        new_width = input_size.width;
        new_height = int(new_width * im_ratio);
    }
    det_scale = float(new_height) / img.rows;
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_width, new_height));
    cv::Mat det_img = cv::Mat::zeros(input_size, img.type());
    // Chú ý: OpenCV (cols=x, rows=y)
    resized_img.copyTo(det_img(cv::Rect(0, 0, new_width, new_height)));
    return det_img;
}

std::vector<FaceObject> predictSCRFD(
    cv::Mat& img,
    const TRTModel& model,
    int inputW = 640, int inputH = 640
){
    nvinfer1::ICudaEngine* engine = model.engine;
    nvinfer1::IExecutionContext* context = model.context;

    int inputIndex = engine->getBindingIndex("input.1");

    
    float det_scale;
    int new_width, new_height;
    cv::Mat prep_img = preprocess_img(img, cv::Size(inputW, inputH), det_scale, new_width, new_height);
    prep_img.convertTo(prep_img, CV_32F);
    cv::cvtColor(prep_img, prep_img, cv::COLOR_BGR2RGB);
    prep_img = (prep_img - 127.5) / 128.0;

    size_t inputSize = 1 * 3 * inputH * inputW * sizeof(float);
    float* inputData;
    CHECK(cudaMalloc((void**)&inputData, inputSize));

    std::vector<float> cpu_input(3 * inputH * inputW);
    std::vector<cv::Mat> chw(3);
    for (int i = 0; i < 3; ++i)
        chw[i] = cv::Mat(inputH, inputW, CV_32F, cpu_input.data() + i * inputH * inputW);
    cv::split(prep_img, chw);

    CHECK(cudaMemcpy(inputData, cpu_input.data(), inputSize, cudaMemcpyHostToDevice));
    const char* output_names[9] = {
        "score_8", "bbox_8", "kps_8",
        "score_16", "bbox_16", "kps_16",
        "score_32", "bbox_32", "kps_32"
    };

    int shapes[9][3] = {
        {1,12800,1}, {1,12800,4}, {1,12800,10},
        {1,3200,1}, {1,3200,4}, {1,3200,10},
        {1,800,1}, {1,800,4}, {1,800,10}
    };

    std::vector<void*> buffers(engine->getNbBindings(), nullptr);
    buffers[inputIndex] = inputData;

    std::vector<float*> output_ptrs(9, nullptr);
    std::vector<size_t> output_sizes(9);
    for (int i = 0; i < 9; ++i) {
        int idx = engine->getBindingIndex(output_names[i]);
        size_t sz = 1;
        for (int k = 0; k < 3; ++k) sz *= shapes[i][k];
        output_sizes[i] = sz;
        CHECK(cudaMalloc((void**)&output_ptrs[i], sz * sizeof(float)));
        buffers[idx] = output_ptrs[i];
    }

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    context->enqueueV2(buffers.data(), stream, nullptr);
    cudaStreamSynchronize(stream);

    std::vector<std::vector<float>> host_outputs(9);
    for (int i = 0; i < 9; ++i) {
        host_outputs[i].resize(output_sizes[i]);
        CHECK(cudaMemcpy(host_outputs[i].data(), output_ptrs[i], output_sizes[i]*sizeof(float), cudaMemcpyDeviceToHost));
    }

    std::vector<std::vector<float>> scores = {host_outputs[0], host_outputs[3], host_outputs[6]};
    std::vector<std::vector<float>> bboxes = {host_outputs[1], host_outputs[4], host_outputs[7]};
    std::vector<std::vector<float>> kpss  = {host_outputs[2], host_outputs[5], host_outputs[8]};

    for (float& v : bboxes[0]) v *= 8;
    for (float& v : kpss[0])  v *= 8;
    for (float& v : bboxes[1]) v *= 16;
    for (float& v : kpss[1])  v *= 16;
    for (float& v : bboxes[2]) v *= 32;
    for (float& v : kpss[2])  v *= 32;

    auto faces = scrfd_postprocess(scores, bboxes, kpss, 0.4, 0.45, inputW, inputH, 5);
    map_faceobjects_to_origin(faces, det_scale, img.cols, img.rows);

    // Cleanup
    CHECK(cudaFree(inputData));
    for (int i = 0; i < 9; ++i) CHECK(cudaFree(output_ptrs[i]));
    cudaStreamDestroy(stream);

    return faces;
}


int main() {
    std::string engine_file = "../checkpoints/scrfd/scrfd_500m_bnkps.engine";
    std::string img_path = "../test/test.jpg";

    TRTModel model = loadModel(engine_file, gLogger);
    // printBindings(model);
    int inputW = 640, inputH = 640;
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "Ảnh không đọc được: " << img_path << std::endl;
        return {};
    }

    auto faces = predictSCRFD(img, model, inputW, inputH);
    for (const auto& face : faces) {
        cv::rectangle(img, face.box, cv::Scalar(0,255,0), 2);
        for (const auto& kp : face.kps)
            cv::circle(img, kp, 2, cv::Scalar(0,0,255), -1);
        // cv::putText... nếu muốn in conf
        std::cout << "Face BBox: ["
              << face.box.x << ", "
              << face.box.y << ", "
              << face.box.width << ", "
              << face.box.height << "]"
              << std::endl;
        }
    cv::imwrite("result.jpg", img);

    // auto engine_data = readEngineFile(engine_file);
    // IRuntime* runtime = createInferRuntime(gLogger);
    // ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    // IExecutionContext* context = engine->createExecutionContext();

    // int inputIndex = engine->getBindingIndex("input.1"); // đổi nếu input tên khác
    
    
    // cv::Mat img = cv::imread(img_path);
    // if (img.empty()) {
    //     std::cerr << "Ảnh không đọc được!" << std::endl;
    //     return -1;
    // }
    // float det_scale;
    // int new_width, new_height;
    // cv::Mat prep_img = preprocess_img(img, cv::Size(inputW, inputH), det_scale, new_width, new_height);

    // prep_img.convertTo(prep_img, CV_32F);
    // cv::cvtColor(prep_img, prep_img, cv::COLOR_BGR2RGB);
    // prep_img = (prep_img - 127.5) / 128.0;

    // size_t inputSize = 1 * 3 * inputH * inputW * sizeof(float);
    // float* inputData;
    // CHECK(cudaMalloc((void**)&inputData, inputSize));

    // std::vector<float> cpu_input(3 * inputH * inputW);
    // std::vector<cv::Mat> chw(3);
    // for (int i = 0; i < 3; ++i)
    //     chw[i] = cv::Mat(inputH, inputW, CV_32F, cpu_input.data() + i * inputH * inputW);
    // cv::split(prep_img, chw);

    // CHECK(cudaMemcpy(inputData, cpu_input.data(), inputSize, cudaMemcpyHostToDevice));

    // const char* output_names[9] = {
    //     "score_8", "bbox_8", "kps_8",
    //     "score_16", "bbox_16", "kps_16",
    //     "score_32", "bbox_32", "kps_32"
    // };

    // int shapes[9][3] = {
    //     {1,12800,1}, {1,12800,4}, {1,12800,10},
    //     {1,3200,1}, {1,3200,4}, {1,3200,10},
    //     {1,800,1}, {1,800,4}, {1,800,10}
    // };
    // std::vector<void*> buffers(engine->getNbBindings(), nullptr);
    // buffers[inputIndex] = inputData;

    // std::vector<float*> output_ptrs(9, nullptr);
    // std::vector<size_t> output_sizes(9);
    // for (int i = 0; i < 9; ++i) {
    //     int idx = engine->getBindingIndex(output_names[i]);
    //     size_t sz = 1;
    //     for (int k = 0; k < 3; ++k) sz *= shapes[i][k];
    //     output_sizes[i] = sz;
    //     CHECK(cudaMalloc((void**)&output_ptrs[i], sz * sizeof(float)));
    //     buffers[idx] = output_ptrs[i];
    // }
    // // Inference
    // cudaStream_t stream;
    // CHECK(cudaStreamCreate(&stream));
    // context->enqueueV2(buffers.data(), stream, nullptr);
    // cudaStreamSynchronize(stream);

    // In ra out của trt (có thể nó khác onnx)
    // int nbBindings = engine->getNbBindings();
    // for (int i = 0; i < nbBindings; ++i) {
    //     const char* name = engine->getBindingName(i);
    //     bool is_input = engine->bindingIsInput(i);
    //     std::cout << (is_input ? "Input" : "Output") << " binding " << i << ": " << name << std::endl;

    //     nvinfer1::Dims dims = engine->getBindingDimensions(i);
    //     std::cout << "  Shape: [";
    //     for (int j = 0; j < dims.nbDims; ++j) {
    //         std::cout << dims.d[j];
    //         if (j < dims.nbDims - 1) std::cout << ", ";
    //     }
    //     std::cout << "]\n";
    // }

    // Copy output to host
    // std::vector<std::vector<float>> host_outputs(9);
    // for (int i = 0; i < 9; ++i) {
    //     host_outputs[i].resize(output_sizes[i]);
    //     CHECK(cudaMemcpy(host_outputs[i].data(), output_ptrs[i], output_sizes[i]*sizeof(float), cudaMemcpyDeviceToHost));
    // }
    // std::vector<std::vector<float>> scores = {host_outputs[0], host_outputs[3], host_outputs[6]};
    // std::vector<std::vector<float>> bboxes = {host_outputs[1], host_outputs[4], host_outputs[7]};
    // std::vector<std::vector<float>> kpss  = {host_outputs[2], host_outputs[5], host_outputs[8]};
    // for (float& v : bboxes[0]) v *= 8;
    // for (float& v : kpss[0])  v *= 8;
    // for (float& v : bboxes[1]) v *= 16;
    // for (float& v : kpss[1])  v *= 16;
    // for (float& v : bboxes[2]) v *= 32;
    // for (float& v : kpss[2])  v *= 32;

    // auto faces = scrfd_postprocess(scores, bboxes, kpss, 0.4, 0.45, inputW, inputH, 5);
    // map_faceobjects_to_origin(faces, det_scale, img.cols, img.rows);
    
}