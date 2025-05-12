#include <iostream>
#include <vector>
#include <string>
#include "gen_data.h"
#include "trainer.h"

int main() {
    std::vector<FunctionType> functions = {
        FunctionType::SQUARE,
        FunctionType::CUBIC_PLUS_LINEAR,
        FunctionType::SINUSOID
    };

    std::vector<ActivationType> activations = {
        ActivationType::SIGMOID,
        ActivationType::SIGMOID_SYMMETRIC,
        ActivationType::SIN_SYMMETRIC
    };

    for (const auto& func : functions) {
        auto target_function = get_function(func);
        std::string func_name = get_function_name(func);
        std::string data_file = func_name + ".data";

        generate_train_data(data_file, 101, -1.0f, 1.0f, target_function);

        for (const auto& act : activations) {
            std::string act_name = get_activation_name(act);
            std::string model_file = "net_" + func_name + "_" + act_name + ".net";

            std::cout << "\n=== Training function: " << func_name
                      << " with activation: " << act_name << " ===\n";

            train_and_test(data_file, model_file, target_function, act);
        }
    }

    return 0;
}
