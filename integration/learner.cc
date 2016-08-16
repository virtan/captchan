#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <math.h>
#include <stdlib.h>
#include <tiny_cnn/tiny_cnn.h>
#include <generator/generator.h>
#include "net_configuration.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

std::vector<vec_t> training_images;
std::vector<vec_t> training_output;
std::vector<vec_t> testing_images;
std::vector<vec_t> testing_output;

tiny_cnn::result test_lenet(network<sequential> &nn) {
    tiny_cnn::result test_result;
    nn.set_netphase(net_phase::test);
    for(size_t i = 0; i < testing_images.size(); ++i) {
        vec_t predicted = nn.predict(testing_images[i]);
        const vec_t &actual = testing_output[i];
        vec_t rounded = round(predicted);
        std::cout << "expected: " << print(actual) << ", got: " << print(predicted) << ", rounded: " << print(rounded) << ", equal: " << (compare(rounded, actual) ? "true" : "false") << std::endl;
        if(compare(rounded, actual)) test_result.num_success++;
        test_result.num_total++;
    }
    return test_result;
}

void train_lenet(int num_epochs) {
    // specify loss-function and learning strategy
    network<sequential> nn;
    adagrad optimizer;

    construct_net(nn);

    std::cout << "start training" << std::endl;
    int current_epoch = 0;
    std::cout << "\nepoch: " << ++current_epoch << "/" << num_epochs << std::endl;

    progress_display disp(training_images.size());
    timer t;
    int minibatch_size = 10;

    // optimizer.alpha *= static_cast<tiny_cnn::float_t>(std::sqrt(minibatch_size));
    optimizer.alpha = 0.5;

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_cnn::result res = test_lenet(nn);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        std::cout << "\nepoch: " << ++current_epoch << "/" << num_epochs << std::endl;
        disp.restart(training_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.fit<mse>(optimizer, training_images, training_output, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    test_lenet(nn).print_detail(std::cout);

    // save networks
    std::ofstream ofs("trained.weights");
    ofs << nn;
}

int main(int argc, char **argv) {
    srandom(time(NULL));
    if(argc != 4) {
        std::cerr << "Usage : " << argv[0] << " <training_size> <testing_size> <epochs>" << std::endl;
        return -1;
    }

    int training_size = std::stoi(argv[1]);
    int testing_size = std::stoi(argv[2]);
    int epochs = std::stoi(argv[3]);

    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::uniform_int_distribution<int> distribution(100000,999999);
    //std::uniform_int_distribution<int> distribution(1, 9);

    {
        std::cout << "constructing training set" << std::endl;
        progress_display disp(training_size);
        timer t;
        size_t size = 0;
        training_images.reserve(training_size);
        training_output.reserve(training_size);
        while(training_size--) {
            int random = distribution(generator);
            std::string str = std::to_string(random);
            vec_t desired_output = code_to_vec(str);
            auto input = generate_vec(str);
            size += input.size() * sizeof(input[0]);
            training_images.emplace_back(vec_t(input.begin(), input.end()));
            vector_rescale(training_images.back());
            training_output.emplace_back(desired_output);
            disp += 1;
        }
        std::cout << "constructed in " << t.elapsed() << "s, total size: " << (((float) size)/1024/1024) << " Mb." << std::endl;

        std::cout << "constructing testing set" << std::endl;
        disp.restart(testing_size);
        t.restart();
        size = 0;
        testing_images.reserve(testing_size);
        testing_output.reserve(testing_size);
        while(testing_size--) {
            int random = distribution(generator);
            std::string str = std::to_string(random);
            vec_t desired_output = code_to_vec(str);
            auto input = generate_vec(str);
            size += input.size() * sizeof(input[0]);
            testing_images.emplace_back(vec_t(input.begin(), input.end()));
            vector_rescale(testing_images.back());
            testing_output.emplace_back(desired_output);
            disp += 1;
        }
        std::cout << "constructed in " << t.elapsed() << "s, total size: " << (((float) size)/1024/1024) << " Mb." << std::endl;
        std::cout << "example: " << training_images.size()/2 << "nth: " << print(training_images[training_images.size()/2]) << ", value " << print(training_output[training_images.size()/2]) << std::endl;
    }

    train_lenet(epochs);
}
