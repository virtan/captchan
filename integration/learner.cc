#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <math.h>
#include <tiny_cnn/tiny_cnn.h>
#include <generator/generator.h>

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

std::vector<vec_t> training_images;
std::vector<vec_t> training_output;
std::vector<vec_t> testing_images;
std::vector<vec_t> testing_output;

void construct_net(network<sequential>& nn) {
    // connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    // construct nets
    nn << convolutional_layer<tan_h>(220, 60, 5, 1, 6)  // C1, 1@220x60-in, 6@216x56-out
       << average_pooling_layer<tan_h>(216, 56, 6, 2)   // S2, 6@216x56-in, 6@108x28-out
       << convolutional_layer<tan_h>(108, 28, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@108x28-in, 16@104x24-in
       << average_pooling_layer<tan_h>(104, 24, 16, 2)  // S4, 16@104x24-in, 16@52x12-out
       << convolutional_layer<tan_h>(52, 12, 5, 16, 120) // C5, 16@52x12-in, 120@48x8-out
       << average_pooling_layer<tan_h>(48, 8, 120, 2)  // S6, 120@48x8-in, 120@24x4-out
       << convolutional_layer<tan_h>(24, 4, 4, 120, 240) // C7, 120@24x4-in, 240@21x1-out
       << fully_connected_layer<tan_h>(5040, 6);       // F8, 5040-in, 6-out
}

vec_t round(const vec_t &s) {
    vec_t r = s;
    for(auto &f : r) f = round(f);
    return r;
}

tiny_cnn::result test_lenet(network<sequential> &nn) {
    tiny_cnn::result test_result;
    nn.set_netphase(net_phase::test);
    for(size_t i = 0; i < testing_images.size(); ++i) {
        vec_t predicted = nn.predict(testing_images[i]);
        const vec_t &actual = testing_output[i];
        if(round(predicted) == actual) test_result.num_success++;
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

    progress_display disp(training_images.size());
    timer t;
    int minibatch_size = 10;

    optimizer.alpha *= static_cast<tiny_cnn::float_t>(std::sqrt(minibatch_size));

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_cnn::result res = test_lenet(nn);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

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
    if(argc != 4) {
        std::cerr << "Usage : " << argv[0] << " <training_size> <testing_size> <epochs>" << std::endl;
        return -1;
    }

    int training_size = std::stoi(argv[1]);
    int testing_size = std::stoi(argv[2]);
    int epochs = std::stoi(argv[3]);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(100000,999999);

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
            std::vector<float> desired_output;
            for(char s : str) if(s) desired_output.push_back((float) (s - '0' + 1));
            auto input = generate_vec(str);
            size += input.size() * sizeof(input[0]);
            training_images.emplace_back(vec_t(input.begin(), input.end()));
            training_output.emplace_back(vec_t(desired_output.begin(), desired_output.end()));
            disp += 1;
        }
        std::cout << "constructed in " << t.elapsed() << "s, total size: " << size << " bytes." << std::endl;

        std::cout << "constructing testing set" << std::endl;
        disp.restart(testing_size);
        t.restart();
        size = 0;
        testing_images.reserve(testing_size);
        testing_output.reserve(testing_size);
        while(testing_size--) {
            int random = distribution(generator);
            std::string str = std::to_string(random);
            std::vector<float> desired_output;
            for(char s : str) if(s) desired_output.push_back((float) (s - '0' + 1));
            auto input = generate_vec(str);
            size += input.size() * sizeof(input[0]);
            testing_images.emplace_back(vec_t(input.begin(), input.end()));
            testing_output.emplace_back(vec_t(desired_output.begin(), desired_output.end()));
            disp += 1;
        }
        std::cout << "constructed in " << t.elapsed() << "s, total size: " << size << " bytes." << std::endl;
    }

    train_lenet(epochs);
}
