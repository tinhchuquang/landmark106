#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cassert>
#include "utils.h"
#include "scrfd.h"
#include <filesystem>
namespace fs = std::filesystem;
using namespace nvinfer1;

struct CropResult {
    cv::Mat cropped;
    int x1;
    int y1;
    int edx1;
    int edy1;
    int size;
};

std::vector<CropResult> load_and_crop_faces(const std::vector<std::string>& img_paths, const TRTModel& scrfd_model) {
    std::vector<CropResult> results;

    for (const auto& path : img_paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "❌ Không đọc được ảnh: " << path << std::endl;
            continue;
        }

        auto faces = predictSCRFD(img, scrfd_model, 640, 640);
        if (faces.empty()) continue;

        auto box = faces[0].box;

        int w = box.width;
        int h = box.height;
        int cx = box.x + w / 2;
        int cy = box.y + h / 2;
        int size = int(std::max(w, h) * 1.1);

        int x1 = cx - size / 2;
        int y1 = cy - size / 2;
        int x2 = x1 + size;
        int y2 = y1 + size;

        int edx1 = std::max(0, -x1);
        int edy1 = std::max(0, -y1);
        int edx2 = std::max(0, x2 - img.cols);
        int edy2 = std::max(0, y2 - img.rows);

        x1 = std::max(0, x1);
        y1 = std::max(0, y1);
        x2 = std::min(img.cols, x2);
        y2 = std::min(img.rows, y2);

        cv::Mat crop = img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
        if (edx1 > 0 || edy1 > 0 || edx2 > 0 || edy2 > 0) {
            cv::copyMakeBorder(crop, crop, edy1, edy2, edx1, edx2,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }

        cv::resize(crop, crop, cv::Size(112, 112));

        results.push_back({crop, x1, y1, edx1, edy1, size});
    }

    return results;
}

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

std::vector<float> runPFLDInference(TRTModel& pfld_model, const std::vector<float>& input_data, int batchSize){
    int inputIndex = pfld_model.engine->getBindingIndex("input");
    int outputIndex = pfld_model.engine->getBindingIndex("output");
    int landmarksIndex = pfld_model.engine->getBindingIndex("landmarks");

    // Set dynamic batch shape
    pfld_model.context->setBindingDimensions(inputIndex, nvinfer1::Dims4(batchSize, 3, 112, 112));
    assert(pfld_model.context->allInputDimensionsSpecified());

    // Tính kích thước bộ nhớ
    size_t inputSize = batchSize * 3 * 112 * 112;
    size_t outputSize = batchSize * 64 * 28 * 28;
    size_t landmarksSize = batchSize * 196;

    // Cấp phát GPU
    void* buffers[3];
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float)));
    CHECK(cudaMalloc(&buffers[landmarksIndex], landmarksSize * sizeof(float)));

    // Copy dữ liệu vào device
    CHECK(cudaMemcpy(buffers[inputIndex], input_data.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));
    // Inference
    pfld_model.context->enqueueV2(buffers, 0, nullptr);

    // Copy kết quả ra host
    std::vector<float> hostOutput(outputSize);
    std::vector<float> hostLandmarks(landmarksSize);
    CHECK(cudaMemcpy(hostOutput.data(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(hostLandmarks.data(), buffers[landmarksIndex], landmarksSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaFree(buffers[landmarksIndex]);

    return hostLandmarks;
}

std::vector<std::vector<cv::Point2f>> postprocessLandMark(
    const std::vector<float>& hostLandmarks,
    const std::vector<CropResult>& cropped_faces,
    const std::vector<std::string>& paths
){
    std::vector<std::vector<cv::Point2f>> all_landmarks;
    int useBatch = std::min<int>(cropped_faces.size(), path.size());
    for (int n = 0; n < useBatch; ++n){
        const CropResult& crop = cropped_faces[n];
        std::vector<cv::Point2f> landmarks;
        for (int j = 0; j < 98; ++j) {
            float x = hostLandmarks[n * 196 + 2 * j] * crop.size - crop.edx1;
            float y = hostLandmarks[n * 196 + 2 * j + 1] * crop.size - crop.edy1;

            int final_x = crop.x1 + static_cast<int>(x + 0.5f);
            int final_y = crop.y1 + static_cast<int>(y + 0.5f);

            landmarks.emplace_back(final_x, final_y);
        }
        all_landmarks.push_back(landmarks);
    }
    return all_landmarks;
}

int main() {
    std::string engine_file_scrfd = "../checkpoints/scrfd/scrfd_500m_bnkps.engine";
    std::string engine_file_pfld = "../checkpoints/pfld/pfld.trt";
    std::string image_paths = "../test";

    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(image_paths)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg" || entry.path().extension() == ".png") {
            paths.push_back(entry.path().string());
        }
    }

    int batchSize = 4;
    TRTModel scrfd_model = loadModel(engine_file_scrfd, gLogger);
    // printBindings(model);
    // auto faces = predictSCRFD(img, model, 640, 640);
    std::vector<CropResult> cropped_faces = load_and_crop_faces(paths, scrfd_model);
    if (cropped_faces.size() < batchSize) {
        std::cerr << "⚠️ Không đủ ảnh cho batch size = " << batchSize << std::endl;
        return -1;
    }
    std::vector<float> input_data(batchSize * 3 * 112 * 112);

    for (int n = 0; n < cropped_faces.size(); ++n) {
        std::vector<float> tmp;
        preprocess_pfld(cropped_faces[n].cropped, tmp);
        std::copy(tmp.begin(), tmp.end(), input_data.begin() + n * 3 * 112 * 112);
    }

    TRTModel pfld_model = loadModel(engine_file_pfld, gLogger);
    assert(pfld_model.context != nullptr);
    // printBindings(pfld_model);

    std::vector<float> hostLandmarks = runPFLDInference(pfld_model, input_data, batchSize);
    auto all_landmarks = postprocessLandmarks(hostLandmarks, cropped_faces, paths);

    for (size_t i = 0; i < all_landmarks.size(); ++i){
        cv::Mat img = cv::imread(paths[i]);
        if (img.empty()) {
            std::cerr << "⚠️ Không đọc được ảnh: " << paths[i] << std::endl;
            continue;
        }
        const auto& landmarks = all_landmarks[i];
        for (const auto& pt : landmarks) {
            cv::circle(img, pt, 2, cv::Scalar(0, 0, 255), -1);
        }

        std::string out_path = "../results/result_" + std::to_string(i) + ".jpg";
        cv::imwrite(out_path, img);
    }

    // Print a few results
    // int usedBatch = std::min<int>(batchSize, cropped_faces.size());
    // for (int n =0; n < usedBatch; ++n){
    //     CropResult crop = cropped_faces[n];
    //     cv::Mat orig_img = cv::imread(paths[n]);
    //     for (int j = 0; j < 98; ++j) {
    //         float x = hostLandmarks[n * 196 + 2 * j] * crop.size - crop.edx1;
    //         float y = hostLandmarks[n * 196 + 2 * j + 1] * crop.size - crop.edy1;

    //         int final_x = crop.x1 + static_cast<int>(x + 0.5f);
    //         int final_y = crop.y1 + static_cast<int>(y + 0.5f);

    //         // std::cout << "  Landmark " << j << ": (" << final_x << ", " << final_y << ")\n";
    //         cv::circle(orig_img, cv::Point(final_x, final_y), 2, cv::Scalar(0, 0, 255), -1);
    //     }
    //     // std::cout << std::endl;
    //     cv::imwrite("../results/result_" + std::to_string(n) + ".jpg", orig_img);
    // }

    destroyModel(pfld_model);
    destroyModel(scrfd_model);

    return 0;
}
