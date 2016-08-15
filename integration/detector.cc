#include <iostream>
#include <fstream>
#include <tiny_cnn/tiny_cnn.h>
#include <generator/generator.h>
#include "net_configuration.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

network<sequential> nn;

void load_net(const std::string &weight_file) {
    construct_net(nn);
    std::ifstream ifs(weight_file.c_str());
    ifs >> nn;
}

bool run_single_test(const std::string &code, std::vector<float> &v) {
    vec_t v_vec(v.begin(), v.end());
    vector_rescale(v_vec);
    vec_t predicted = nn.predict(v_vec);
    vec_t actual = code_to_vec(code);
    vec_t rounded = round(predicted);
    // std::cout << "expected: " << print(actual) << ", got: " << print(predicted) << ", rounded: " << print(rounded) << ", equal: " << (compare(rounded, actual) ? "true" : "false") << std::endl;
    return compare(rounded, actual);
}

void run_test_set(int tests) {
    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::uniform_int_distribution<int> distribution(100000,999999);
    //std::uniform_int_distribution<int> distribution(1, 9);
    progress_display disp(tests);
    timer t;
    tiny_cnn::result test_result;
    while(tests--) {
        int random = distribution(generator);
        std::string code = std::to_string(random);
        auto input = generate_vec(code);
        bool correct = run_single_test(code, input);
        if(correct) test_result.num_success++;
        test_result.num_total++;
        disp += 1;
    }
    std::cout << "tested in " << t.elapsed() << "s, ";
    std::cout << "success: " << test_result.num_success << "/" << test_result.num_total << std::endl;
}

int main(int argc, char **argv) {
    if(argc != 3 && argc != 4) {
        std::cout << "Usage: " << argv[0] << " <weights_file> <tests>\n";
        std::cout << "    or " << argv[0] << " <weights_file> <code> <output_png_filename>\n";
        return 1;
    }
    std::string weight_file = argv[1];
    load_net(weight_file);
    if(argc == 3) {
        int tests = std::stoi(argv[2]);
        run_test_set(tests);
    } else {
        std::string code = argv[2];
        if(code.size() != 6) {
            std::cerr << "Warning: network is trained to recognize 6-digit captchas\n";
        }
        std::string output_png_filename = argv[3];
        captcha c = generate(code);
        std::ofstream output;
        output.open(output_png_filename.c_str());
        output << c.png;
        output.close();
        bool correct = run_single_test(code, c.vec);
        std::cout << (correct ? "correct prediction" : "wrong prediction") << std::endl;
    }
    return 0;
}
