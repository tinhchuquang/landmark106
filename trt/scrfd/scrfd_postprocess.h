#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>

struct FaceObject {
    float conf;
    cv::Rect2f box;
    std::vector<cv::Point2f> kps;
};

inline std::vector<cv::Point2f> generate_anchor_centers(int height, int width, int stride, int num_anchors = 2) {
    std::vector<cv::Point2f> anchor_centers;
    anchor_centers.reserve(height * width * num_anchors);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int n = 0; n < num_anchors; ++n)    // lặp lại cho mỗi anchor
                anchor_centers.emplace_back(x * stride, y * stride);
    return anchor_centers;
}

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

inline std::vector<FaceObject> scrfd_postprocess(
    const std::vector<std::vector<float>>& scores,  // {score8, score16, score32}
    const std::vector<std::vector<float>>& bboxes,  // {bbox8, bbox16, bbox32}
    const std::vector<std::vector<float>>& kpss,    // {kps8, kps16, kps32}
    float score_thresh, float nms_thresh,
    int inputW, int inputH,
    int num_kps = 5
) {
    std::vector<int> strides = {8, 16, 32};
    std::vector<FaceObject> proposals;
    for (int l = 0; l < 3; ++l) {
        int stride = strides[l];
        int feat_h = inputH / stride;
        int feat_w = inputW / stride;
        int K = feat_h * feat_w;
        auto anchor_centers = generate_anchor_centers(feat_h, feat_w, stride);
        // std::cout << "anchor_centers shape: (" << anchor_centers.size() << ", 2)" << std::endl;

        // // In 5 phần tử đầu
        // // std::cout << "First 5 anchor centers: " << std::endl;
        // for (int i = 0; i < 5 && i < anchor_centers.size(); ++i) {
        //     std::cout << "[" << anchor_centers[i].x << ", " << anchor_centers[i].y << "]" << std::endl;
        // }
        // Nhân stride cho bbox/kps trước khi truyền vào!
        for (int i = 0; i < K; ++i) {
            float conf = scores[l][i];
            if (conf < score_thresh) continue;
            const cv::Point2f& center = anchor_centers[i];
            const float* dbox = &bboxes[l][i * 4];
            float x1 = center.x - dbox[0];
            float y1 = center.y - dbox[1];
            float x2 = center.x + dbox[2];
            float y2 = center.y + dbox[3];
            cv::Rect2f box(x1, y1, x2 - x1, y2 - y1);
            std::vector<cv::Point2f> kps;
            if (!kpss.empty() && kpss[l].size() > 0) {
                const float* kps_ptr = &kpss[l][i * num_kps * 2];
                for (int k = 0; k < num_kps; ++k) {
                    float px = center.x + kps_ptr[2 * k + 0];
                    float py = center.y + kps_ptr[2 * k + 1];
                    kps.emplace_back(px, py);
                }
            }
            proposals.push_back({conf, box, kps});
        }
    }
    // NMS cuối cùng
    std::vector<FaceObject> results;
    nms(proposals, results, nms_thresh);
    return results;
}

inline void map_faceobjects_to_origin(
    std::vector<FaceObject>& faces,
    float det_scale,
    int origin_w,
    int origin_h
) {
    for (auto& face : faces) {
        face.box.x = std::clamp(face.box.x / det_scale, 0.f, float(origin_w - 1));
        face.box.y = std::clamp(face.box.y / det_scale, 0.f, float(origin_h - 1));
        face.box.width = std::clamp(face.box.width / det_scale, 0.f, float(origin_w - face.box.x));
        face.box.height = std::clamp(face.box.height / det_scale, 0.f, float(origin_h - face.box.y));
        for (auto& kp : face.kps) {
            kp.x = std::clamp(kp.x / det_scale, 0.0f, float(origin_w-1));
            kp.y = std::clamp(kp.y / det_scale, 0.0f, float(origin_h-1));
        }
    }
}
