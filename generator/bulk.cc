#include <iostream>
#include <string>
#include <random>
#include <zlib.h>
#include <tiny_dnn/tiny_dnn.h>

#include "generator.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

int main(int argc, char **argv) {
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file_prefix> <amount>\n";
        exit(1);
    }
    std::string file_prefix = argv[1];
    int amount = atoi(argv[2]);

    std::default_random_engine generator;
    generator.seed(time(NULL));
    std::uniform_int_distribution<int> distribution(100000,999999);

    gzFile values = gzopen((file_prefix + "_values.gz").c_str(), "wb9");
    gzFile data = gzopen((file_prefix + "_data.gz").c_str(), "wb9");
    if(!values || !data) {
        std::cerr << "Can't open files for writing\n";
        exit(1);
    }

    progress_display disp(amount);
    std::string value;
    std::string vecs;
    while(amount--) {
        int random = distribution(generator);
        std::string str = std::to_string(random);
        auto vec = generate_vec(str);

        value.clear();
        for(char s : str) if(s) value.append(1, ' ').append(1, s).append(".0");
        gzwrite(values, value.data(), value.size());

        vecs.clear();
        for(float f : vec) vecs.append(1, ' ').append(std::to_string(f));
        gzwrite(data, vecs.data(), vecs.size());

        disp += 1;
    }

    gzclose(values);
    gzclose(data);
    return 0;
}
