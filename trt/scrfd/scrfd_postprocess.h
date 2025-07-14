#pragma once
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>

struct FaceObject {
    float conf;
    cv::Rect box;
    std::vector<cv::Point2f> kps;
};

inline void map_faceobjects_to_origin(
    std::vector<FaceObject>& faces,
    float det_scale,
    int new_width,
    int new_height,
    int origin_w,
    int origin_h
) {
    for (auto& face : faces) {
        // Trừ đi pad (ở đây pad_x, pad_y đều là 0 vì luôn pad góc phải/dưới)
        face.box.x = std::clamp(int(face.box.x * 1.0 / det_scale), 0, origin_w-1);
        face.box.y = std::clamp(int(face.box.y * 1.0 / det_scale), 0, origin_h-1);
        face.box.width = std::clamp(int(face.box.width * 1.0 / det_scale), 0, origin_w - face.box.x);
        face.box.height = std::clamp(int(face.box.height * 1.0 / det_scale), 0, origin_h - face.box.y);

        for (auto& kp : face.kps) {
            kp.x = std::clamp(float(kp.x / det_scale), 0.0f, float(origin_w-1));
            kp.y = std::clamp(float(kp.y / det_scale), 0.0f, float(origin_h-1));
        }
    }
}

// NMS truyền thống cho box
inline void nms(const std::vector<FaceObject>& dets, std::vector<FaceObject>& out, float nms_thresh) {
    out.clear();
    std::vector<size_t> idxs(dets.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), [&dets](size_t a, size_t b) {
        return dets[a].conf > dets[b].conf;
    });
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < idxs.size(); ++i) {
        if (suppressed[idxs[i]]) continue;
        out.push_back(dets[idxs[i]]);
        for (size_t j = i + 1; j < idxs.size(); ++j) {
            if (suppressed[idxs[j]]) continue;
            auto& b1 = dets[idxs[i]].box;
            auto& b2 = dets[idxs[j]].box;
            float inter = (b1 & b2).area();
            float uni = b1.area() + b2.area() - inter;
            if (uni <= 0) continue;
            float iou = inter / uni;
            if (iou > nms_thresh) suppressed[idxs[j]] = true;
        }
    }
}

// Postprocess SCRFD multi-head (score, bbox, kps) cho 3 scale
inline std::vector<FaceObject> scrfd_postprocess(
    const std::vector<std::vector<float>>& scores,
    const std::vector<std::vector<float>>& bboxes,
    const std::vector<std::vector<float>>& kpss,
    float score_thresh, float nms_thresh,
    int inputW, int inputH,
    int num_kps = 5
) {
    // Số anchor mỗi level SCRFD thường là: {12800, 3200, 800}
    int strides[3] = {8, 16, 32};
    int featW[3] = {inputW / 8, inputW / 16, inputW / 32};
    int featH[3] = {inputH / 8, inputH / 16, inputH / 32};
    int kps_dim = num_kps * 2; // (x1,y1,...xn,yn)

    std::vector<FaceObject> proposals;
    for (int l = 0; l < 3; ++l) {
        int num_anchors = featH[l] * featW[l];
        const float* score_ptr = scores[l].data();
        const float* bbox_ptr = bboxes[l].data();
        const float* kps_ptr = kpss[l].data();

        for (int i = 0; i < num_anchors; ++i) {
            float conf = score_ptr[i];
            if (conf < score_thresh) continue;

            int ax = i % featW[l];
            int ay = i / featW[l];
            int cx = ax * strides[l] + strides[l] / 2;
            int cy = ay * strides[l] + strides[l] / 2;

            float dx = bbox_ptr[i * 4 + 0];
            float dy = bbox_ptr[i * 4 + 1];
            float dw = bbox_ptr[i * 4 + 2];
            float dh = bbox_ptr[i * 4 + 3];

            float x1 = cx - dx;
            float y1 = cy - dy;
            float x2 = cx + dw;
            float y2 = cy + dh;
            cv::Rect box(round(x1), round(y1), round(x2 - x1), round(y2 - y1));

            std::vector<cv::Point2f> kps;
            for (int k = 0; k < num_kps; ++k) {
                float kpx = cx + kps_ptr[i * kps_dim + 2 * k + 0];
                float kpy = cy + kps_ptr[i * kps_dim + 2 * k + 1];
                kps.emplace_back(kpx, kpy);
            }
            proposals.push_back({conf, box, kps});
        }
    }
    std::vector<FaceObject> results;
    nms(proposals, results, nms_thresh);
    return results;
}