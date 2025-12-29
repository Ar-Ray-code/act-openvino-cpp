/*
 * Copyright (c) Ar-Ray-code 2025
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ACT_INFERENCE_H
#define ACT_INFERENCE_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace act {

class ACTInference {
public:
    explicit ACTInference(const std::string& model_path, const std::string& device = "CPU");
    ~ACTInference() = default;

    std::vector<std::vector<float>> inference(
        const std::vector<float> &state,
        const cv::Mat &image
    );

    std::vector<float> preprocessImage(const cv::Mat &image);

    void printModelInfo();

    // Get input dimensions from model
    size_t getStateSize() const;
    int getImageHeight() const;
    int getImageWidth() const;
    int getImageChannels() const;

    // Create dummy state vector with correct size
    std::vector<float> createDummyState() const;

    // Create dummy image with correct dimensions
    cv::Mat createDummyImage() const;

private:
    ov::Core core_;
    std::shared_ptr<ov::Model> model_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<ov::Shape> input_shapes_;
    std::vector<ov::Shape> output_shapes_;

    const std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    const std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
};

} // namespace act

#endif // ACT_INFERENCE_H
