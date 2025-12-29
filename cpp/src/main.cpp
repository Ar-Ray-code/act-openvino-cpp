/*
 * Copyright (c) Ar-Ray-code 2025
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <iomanip>

#include "act_inference.h"

void printActions(const std::vector<std::vector<float>>& actions, int num_steps = 10) {
    std::cout << "\nPredicted Actions (first " << num_steps << " steps):" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    for (size_t i = 0; i < std::min(static_cast<size_t>(num_steps), actions.size()); i++) {
        std::cout << "  Step " << std::setw(3) << i << ": [";
        for (size_t j = 0; j < actions[i].size(); j++) {
            std::cout << std::setw(8) << actions[i][j];
            if (j < actions[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "  ... (" << actions.size() << " total steps)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "ACT Model Inference with OpenVINO" << std::endl;
    std::cout << "========================================" << std::endl;

    // Parse command line arguments
    std::string model_path = "../output/act_model_fp16.xml";
    std::string image_path = "";
    std::string device = "CPU";

    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        image_path = argv[2];
    }
    if (argc >= 4) {
        device = argv[3];
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Model path: " << model_path << std::endl;
    std::cout << "  Image path: " << (image_path.empty() ? "dummy data" : image_path) << std::endl;
    std::cout << "  Device: " << device << std::endl;
    std::cout << std::endl;

    try {
        // Create inference engine
        act::ACTInference inference(model_path, device);
        inference.printModelInfo();

        // Get model input dimensions
        std::cout << "\nModel input requirements:" << std::endl;
        std::cout << "  State size: " << inference.getStateSize() << std::endl;
        std::cout << "  Image size: " << inference.getImageWidth() << "x" << inference.getImageHeight()
                  << " (" << inference.getImageChannels() << " channels)" << std::endl;

        // Prepare input data - use model dimensions
        std::vector<float> state;
        if (!image_path.empty()) {
            // For real images, create a simple state vector
            state = inference.createDummyState();
        } else {
            // Create dummy state from model
            state = inference.createDummyState();
        }

        cv::Mat image;
        if (!image_path.empty()) {
            // Load image from file
            cv::Mat loaded_image = cv::imread(image_path);
            if (loaded_image.empty()) {
                std::cerr << "Failed to load image: " << image_path << std::endl;
                return 1;
            }
            std::cout << "\nLoaded image: " << loaded_image.size() << std::endl;

            // Resize to model's expected dimensions
            int target_width = inference.getImageWidth();
            int target_height = inference.getImageHeight();
            if (loaded_image.cols != target_width || loaded_image.rows != target_height) {
                cv::resize(loaded_image, image, cv::Size(target_width, target_height));
                std::cout << "Resized image to: " << image.size() << std::endl;
            } else {
                image = loaded_image;
            }
        } else {
            // Create dummy image using model dimensions
            image = inference.createDummyImage();
            std::cout << "\nCreated dummy image: " << image.size() << std::endl;
        }

        std::cout << "\nInput:" << std::endl;
        std::cout << "  State: [";
        for (size_t i = 0; i < state.size(); i++) {
            std::cout << state[i];
            if (i < state.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Image: " << image.size() << " (" << image.channels() << " channels)" << std::endl;

        // Run inference
        std::cout << "\nRunning inference..." << std::endl;
        auto actions = inference.inference(state, image);

        // Print results
        printActions(actions);

        std::cout << "\nInference completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
