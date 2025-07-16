#pragma once

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "utils.h"
#include "scrfd_postprocess.h"

// Tiền xử lý ảnh SCRFD
cv::Mat preprocess_img(const cv::Mat& img, cv::Size input_size, float& det_scale, int& new_width, int& new_height);

// Chạy inference SCRFD bằng TensorRT, trả về danh sách face objects
std::vector<FaceObject> predictSCRFD(
    cv::Mat& img,
    const TRTModel& model,
    int inputW = 640,
    int inputH = 640
);
