/*
 * Copyright (c) Ar-Ray-code 2025
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>


struct BenchmarkStats {
    std::string device;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    double p50_ms;
    double p95_ms;
    double p99_ms;
    double fps;
};

// Calculate percentile
double percentile(std::vector<double>& data, double p) {
    size_t n = data.size();
    double index = (p / 100.0) * (n - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper) {
        return data[lower];
    }

    double weight = index - lower;
    return data[lower] * (1 - weight) + data[upper] * weight;
}

// Create dummy inputs based on model input shapes
std::pair<ov::Tensor, ov::Tensor> createDummyInputs(const ov::Shape& state_shape, const ov::Shape& image_shape) {
    // State tensor: dynamically sized based on model
    ov::Tensor state_tensor(ov::element::f32, state_shape);
    float* state_data = state_tensor.data<float>();
    size_t state_size = state_tensor.get_size();
    for (size_t i = 0; i < state_size; i++) {
        state_data[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Image tensor: dynamically sized based on model
    ov::Tensor image_tensor(ov::element::f32, image_shape);
    float* image_data = image_tensor.data<float>();

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    // Extract dimensions from shape: [batch, channels, height, width]
    int channels = static_cast<int>(image_shape[1]);
    int height = static_cast<int>(image_shape[2]);
    int width = static_cast<int>(image_shape[3]);

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = c * height * width + h * width + w;
                float pixel = static_cast<float>(std::rand() % 256) / 255.0f;
                image_data[idx] = (pixel - mean[c]) / std[c];
            }
        }
    }

    return std::make_pair(state_tensor, image_tensor);
}

BenchmarkStats benchmarkDevice(const std::string& model_path, const std::string& device, int num_iterations) {
    ov::Core core;

    std::cout << "\nAvailable devices: ";
    for (const auto& dev : core.get_available_devices()) {
        std::cout << dev << " ";
    }
    std::cout << std::endl;

    // Read model
    std::cout << "\nLoading model: " << model_path << std::endl;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    // Compile model
    std::cout << "Compiling model for " << device << "..." << std::endl;
    ov::CompiledModel compiled_model = core.compile_model(model, device);

    // Get input/output info
    auto inputs = compiled_model.inputs();
    auto outputs = compiled_model.outputs();

    std::cout << "\nModel Inputs:" << std::endl;
    for (size_t i = 0; i < inputs.size(); i++) {
        std::cout << "  Input " << i << ": " << inputs[i].get_any_name() << " ";
        std::cout << inputs[i].get_shape() << std::endl;
    }

    std::cout << "\nModel Outputs:" << std::endl;
    for (size_t i = 0; i < outputs.size(); i++) {
        std::cout << "  Output " << i << ": " << outputs[i].get_any_name() << " ";
        std::cout << outputs[i].get_shape() << std::endl;
    }

    // Create inference request
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input shapes from model
    ov::Shape state_shape = inputs[0].get_shape();
    ov::Shape image_shape = inputs[1].get_shape();

    // Prepare dummy inputs based on model shapes
    auto [state_tensor, image_tensor] = createDummyInputs(state_shape, image_shape);

    // Set inputs
    infer_request.set_input_tensor(0, state_tensor);
    infer_request.set_input_tensor(1, image_tensor);

    // Warmup
    std::cout << "\nWarmup (10 iterations)..." << std::endl;
    for (int i = 0; i < 10; i++) {
        infer_request.infer();
    }

    // Benchmark
    std::cout << "\nRunning benchmark (" << num_iterations << " iterations)..." << std::endl;
    std::vector<double> latencies;
    latencies.reserve(num_iterations);

    for (int i = 0; i < num_iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        infer_request.infer();
        auto end = std::chrono::high_resolution_clock::now();

        double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(latency_ms);

        if ((i + 1) % 10 == 0) {
            std::cout << "  Iteration " << std::setw(3) << (i + 1) << "/" << num_iterations
                 << ": " << std::fixed << std::setprecision(2) << latency_ms << " ms" << std::endl;
        }
    }

    // Calculate statistics
    std::vector<double> sorted_latencies = latencies;
    std::sort(sorted_latencies.begin(), sorted_latencies.end());

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double mean = sum / latencies.size();

    double sq_sum = 0.0;
    for (double lat : latencies) {
        sq_sum += (lat - mean) * (lat - mean);
    }
    double std_dev = std::sqrt(sq_sum / latencies.size());

    BenchmarkStats stats;
    stats.device = device;
    stats.mean_ms = mean;
    stats.std_ms = std_dev;
    stats.min_ms = sorted_latencies.front();
    stats.max_ms = sorted_latencies.back();
    stats.p50_ms = percentile(sorted_latencies, 50);
    stats.p95_ms = percentile(sorted_latencies, 95);
    stats.p99_ms = percentile(sorted_latencies, 99);
    stats.fps = 1000.0 / mean;

    // Print results
    std::cout << "\n" << std::string(80, '-') << std::endl;
    std::cout << "Results for " << device << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Latency Statistics (ms):" << std::endl;
    std::cout << "  Mean:   " << std::setw(8) << stats.mean_ms << std::endl;
    std::cout << "  Std:    " << std::setw(8) << stats.std_ms << std::endl;
    std::cout << "  Min:    " << std::setw(8) << stats.min_ms << std::endl;
    std::cout << "  Max:    " << std::setw(8) << stats.max_ms << std::endl;
    std::cout << "  P50:    " << std::setw(8) << stats.p50_ms << std::endl;
    std::cout << "  P95:    " << std::setw(8) << stats.p95_ms << std::endl;
    std::cout << "  P99:    " << std::setw(8) << stats.p99_ms << std::endl;
    std::cout << "\nThroughput:" << std::endl;
    std::cout << "  FPS:    " << std::setw(8) << stats.fps << std::endl;

    // Check output
    ov::Tensor output_tensor = infer_request.get_output_tensor(0);
    float* output_data = output_tensor.data<float>();
    size_t output_size = output_tensor.get_size();

    double output_sum = 0.0;
    float output_min = output_data[0];
    float output_max = output_data[0];

    for (size_t i = 0; i < output_size; i++) {
        output_sum += output_data[i];
        output_min = std::min(output_min, output_data[i]);
        output_max = std::max(output_max, output_data[i]);
    }

    double output_mean = output_sum / output_size;

    double output_sq_sum = 0.0;
    for (size_t i = 0; i < output_size; i++) {
        output_sq_sum += (output_data[i] - output_mean) * (output_data[i] - output_mean);
    }
    double output_std = std::sqrt(output_sq_sum / output_size);

    std::cout << "\nOutput Shape: " << output_tensor.get_shape() << std::endl;
    std::cout << "Output Stats:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Mean:   " << std::setw(8) << output_mean << std::endl;
    std::cout << "  Std:    " << std::setw(8) << output_std << std::endl;
    std::cout << "  Min:    " << std::setw(8) << output_min << std::endl;
    std::cout << "  Max:    " << std::setw(8) << output_max << std::endl;

    return stats;
}

int main(int argc, char* argv[]) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::string model_path = "../output/openvino/act_model.xml";
    std::string device_filter = "";  // Empty = run both CPU and GPU
    int num_iterations = 100;

    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        device_filter = argv[2];  // "CPU" or "GPU" or empty for both
    }
    if (argc >= 4) {
        num_iterations = std::stoi(argv[3]);
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Model path: " << model_path << std::endl;
    std::cout << "  Device filter: " << (device_filter.empty() ? "Both (CPU and GPU)" : device_filter) << std::endl;
    std::cout << "  Iterations: " << num_iterations << std::endl;

    try {
        std::ifstream model_file(model_path);
        if (!model_file.good()) {
            std::cerr << "\nError: Model file not found: " << model_path << std::endl;
            std::cerr << "Please convert ONNX to OpenVINO IR first:" << std::endl;
            return 1;
        }

        bool run_cpu = device_filter.empty() || device_filter == "CPU";
        bool run_gpu = device_filter.empty() || device_filter == "GPU";

        BenchmarkStats cpu_stats, gpu_stats;

        if (run_cpu) {
            cpu_stats = benchmarkDevice(model_path, "CPU", num_iterations);
        }

        if (run_gpu) {
            gpu_stats = benchmarkDevice(model_path, "GPU", num_iterations);
        }
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
