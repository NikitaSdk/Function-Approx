#include "trainer.h"
#include <fann.h>
#include <fann_cpp.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

std::function<float(float)> get_function(FunctionType type) {
    switch (type) {
        case FunctionType::SQUARE: return [](float x) { return x * x; };
        case FunctionType::CUBIC_PLUS_LINEAR: return [](float x) { return x * x * x + 2 * x; };
        case FunctionType::SINUSOID: return [](float x) { return std::sin(x); };
        default: return [](float) { return 0.0f; };
    }
}

std::string get_function_name(FunctionType type) {
    switch (type) {
        case FunctionType::SQUARE: return "square";
        case FunctionType::CUBIC_PLUS_LINEAR: return "cubic";
        case FunctionType::SINUSOID: return "sin";
        default: return "unknown";
    }
}

std::string get_activation_name(ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID: return "sigmoid";
        case ActivationType::SIGMOID_SYMMETRIC: return "symmetric_sigmoid";
        case ActivationType::SIN_SYMMETRIC: return "sin_symmetric";
        default: return "unknown";
    }
}

void train_and_test(const std::string& train_file,
                    const std::string& model_file,
                    std::function<float(float)> target_function,
                    ActivationType activation) {
    FANN::neural_net ann;
    ann.create_standard(3, 1, 10, 1);
    ann.set_learning_rate(0.01f);

    switch (activation) {
        case ActivationType::SIGMOID:
            ann.set_activation_function_hidden(FANN::SIGMOID);
            break;
        case ActivationType::SIGMOID_SYMMETRIC:
            ann.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC);
            break;
        case ActivationType::SIN_SYMMETRIC:
            ann.set_activation_function_hidden(FANN::SIN_SYMMETRIC);
            break;
    }

    ann.set_activation_function_output(FANN::LINEAR);
    ann.set_training_algorithm(FANN::TRAIN_INCREMENTAL);

    FANN::training_data data;
    if (!data.read_train_from_file(train_file)) {
        std::cerr << "Failed to read training file: " << train_file << std::endl;
        return;
    }

    for (unsigned int i = 1; i <= 3000; ++i) {
        float mse = ann.train_epoch(data);
        if (i % 500 == 0 || i == 1 || i == 3000)
            std::cout << "Iter " << i << ", MSE = " << mse << std::endl;
    }

    ann.save(model_file);

    std::string results_dir = "results";
    fs::create_directory(results_dir);

    std::string csv_filename = results_dir + "/" +
        get_function_name_from_file(train_file) + "_" +
        get_activation_name(activation) + ".csv";

    std::ofstream fout(csv_filename);
    if (!fout.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_filename << std::endl;
        return;
    }

    fout << "x,predicted,expected\n";
    std::cout << "\n--- Predictions ---\n";

    for (int i = 0; i <= 10; ++i) {
        float x = static_cast<float>(i) / 5.0f - 1.0f;
        fann_type input[1] = {x};
        fann_type* output = ann.run(input);
        float predicted = output[0];
        float expected = target_function(x);

        std::cout << "x = " << x
                  << ", predicted = " << predicted
                  << ", expected = " << expected << std::endl;

        fout << x << "," << predicted << "," << expected << "\n";
    }

    fout.close();
    std::cout << "Saved predictions to: " << csv_filename << std::endl;
}

std::string get_function_name_from_file(const std::string& filename) {
    size_t slash_pos = filename.find_last_of("/\\");
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos == std::string::npos) dot_pos = filename.length();
    return filename.substr(slash_pos == std::string::npos ? 0 : slash_pos + 1,
                           dot_pos - (slash_pos == std::string::npos ? 0 : slash_pos + 1));
}
