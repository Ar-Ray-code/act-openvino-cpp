/*
 * Copyright (c) Ar-Ray-code 2025
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "act_inference.h"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace act {

ACTInference::ACTInference(const std::string& model_path, const std::string& device) {
    std::cout << "Loading OpenVINO model: " << model_path << std::endl;

    model_ = core_.read_model(model_path);

    std::cout << "Compiling model for device: " << device << std::endl;
    compiled_model_ = core_.compile_model(model_, device);

    infer_request_ = compiled_model_.create_infer_request();

    auto inputs = compiled_model_.inputs();
    input_names_.reserve(inputs.size());
    input_shapes_.reserve(inputs.size());

    for (const auto& input : inputs) {
        input_names_.push_back(input.get_any_name());
        input_shapes_.push_back(input.get_shape());
    }

    auto outputs = compiled_model_.outputs();
    output_names_.reserve(outputs.size());
    output_shapes_.reserve(outputs.size());

    for (const auto& output : outputs) {
        output_names_.push_back(output.get_any_name());
        output_shapes_.push_back(output.get_shape());
    }

    std::cout << "Model loaded successfully!" << std::endl;
}

void ACTInference::printModelInfo() {
    std::cout << "\n=== Model Information ===" << std::endl;

    std::cout << "\nInputs (" << input_names_.size() << "):" << std::endl;
    for (size_t i = 0; i < input_names_.size(); i++) {
        std::cout << "  [" << i << "] " << input_names_[i] << ": ";
        std::cout << input_shapes_[i] << std::endl;
    }

    std::cout << "\nOutputs (" << output_names_.size() << "):" << std::endl;
    for (size_t i = 0; i < output_names_.size(); i++) {
        std::cout << "  [" << i << "] " << output_names_[i] << ": ";
        std::cout << output_shapes_[i] << std::endl;
    }
    std::cout << "========================\n" << std::endl;
}

std::vector<float> ACTInference::preprocessImage(const cv::Mat& image) {
    // Expected input: RGB image
    // Output: CHW format, normalized with ImageNet stats

    if (input_shapes_.size() < 2 || input_shapes_[1].size() != 4) {
        throw std::runtime_error("Invalid image input shape in model");
    }

    int channels = static_cast<int>(input_shapes_[1][1]);
    int height = static_cast<int>(input_shapes_[1][2]);
    int width = static_cast<int>(input_shapes_[1][3]);

    cv::Mat resized;
    if (image.size() != cv::Size(width, height)) {
        cv::resize(image, resized, cv::Size(width, height));
    } else {
        resized = image.clone();
    }

    cv::Mat rgb;
    if (image.channels() == 3) {
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = resized;
    }

    cv::Mat float_image;
    rgb.convertTo(float_image, CV_32F, 1.0 / 255.0);

    std::vector<float> input_data(channels * height * width);

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = c * height * width + h * width + w;
                float pixel = float_image.at<cv::Vec3f>(h, w)[c];
                input_data[idx] = (pixel - mean_[c]) / std_[c];
            }
        }
    }

    return input_data;
}

std::vector<std::vector<float>> ACTInference::inference(
    const std::vector<float>& state,
    const cv::Mat& image
) {
    if (input_shapes_.size() < 1 || input_shapes_[0].size() < 2) {
        throw std::runtime_error("Invalid state input shape in model");
    }

    size_t expected_state_size = static_cast<size_t>(input_shapes_[0][1]);
    if (state.size() != expected_state_size) {
        throw std::runtime_error("State vector size (" + std::to_string(state.size()) +
                                 ") does not match model input size (" + std::to_string(expected_state_size) + ")");
    }

    auto image_data = preprocessImage(image);

    // Create input tensors
    // Input 0: observation.state
    ov::Tensor state_tensor(ov::element::f32, input_shapes_[0]);
    float* state_ptr = state_tensor.data<float>();
    std::copy(state.begin(), state.end(), state_ptr);

    // Input 1: observation.images.front
    ov::Tensor image_tensor(ov::element::f32, input_shapes_[1]);
    float* image_ptr = image_tensor.data<float>();
    std::copy(image_data.begin(), image_data.end(), image_ptr);

    infer_request_.set_input_tensor(0, state_tensor);
    infer_request_.set_input_tensor(1, image_tensor);

    auto start = std::chrono::high_resolution_clock::now();
    infer_request_.infer();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

    ov::Tensor output_tensor = infer_request_.get_output_tensor(0);
    float* output_data = output_tensor.data<float>();
    ov::Shape output_shape = output_tensor.get_shape();

    // Output shape: [batch, chunk_size, action_dim]
    int chunk_size = static_cast<int>(output_shape[1]);
    int action_dim = static_cast<int>(output_shape[2]);

    std::vector<std::vector<float>> actions(chunk_size, std::vector<float>(action_dim));
    for (int i = 0; i < chunk_size; i++) {
        for (int j = 0; j < action_dim; j++) {
            actions[i][j] = output_data[i * action_dim + j];
        }
    }

    return actions;
}

size_t ACTInference::getStateSize() const {
    if (input_shapes_.size() < 1 || input_shapes_[0].size() < 2) {
        throw std::runtime_error("Invalid state input shape in model");
    }
    return static_cast<size_t>(input_shapes_[0][1]);
}

int ACTInference::getImageHeight() const {
    if (input_shapes_.size() < 2 || input_shapes_[1].size() != 4) {
        throw std::runtime_error("Invalid image input shape in model");
    }
    return static_cast<int>(input_shapes_[1][2]);
}

int ACTInference::getImageWidth() const {
    if (input_shapes_.size() < 2 || input_shapes_[1].size() != 4) {
        throw std::runtime_error("Invalid image input shape in model");
    }
    return static_cast<int>(input_shapes_[1][3]);
}

int ACTInference::getImageChannels() const {
    if (input_shapes_.size() < 2 || input_shapes_[1].size() != 4) {
        throw std::runtime_error("Invalid image input shape in model");
    }
    return static_cast<int>(input_shapes_[1][1]);
}

std::vector<float> ACTInference::createDummyState() const {
    size_t state_size = getStateSize();
    std::vector<float> state(state_size);
    for (size_t i = 0; i < state_size; i++) {
        state[i] = static_cast<float>(i + 1) / static_cast<float>(state_size + 1);
    }
    return state;
}

cv::Mat ACTInference::createDummyImage() const {
    int height = getImageHeight();
    int width = getImageWidth();
    int channels = getImageChannels();

    cv::Mat image = cv::Mat::zeros(height, width, channels == 3 ? CV_8UC3 : CV_8UC1);

    if (channels == 3) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    static_cast<uint8_t>((i * 255) / height),
                    static_cast<uint8_t>((j * 255) / width),
                    128
                );
            }
        }
    } else {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                image.at<uint8_t>(i, j) = static_cast<uint8_t>((i + j) * 255 / (height + width));
            }
        }
    }

    return image;
}

} // namespace act
