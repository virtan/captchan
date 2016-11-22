#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <random>
#include <time.h>
#include <sys/time.h>

#include "progress.h"
#include "generator.h"

int main(int argc, char **argv) {
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file_prefix> <amount>\n";
        exit(1);
    }

    std::string file_prefix = argv[1];
    int amount = atoi(argv[2]);

    std::default_random_engine generator;
#ifdef __linux
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    generator.seed(ts.tv_nsec);
#else
    struct timeval tv;
    gettimeofday(&tv, 0);
    generator.seed(tv.tv_usec);
#endif

    std::uniform_int_distribution<int> distribution(100000,999999);

    progress disp; // amount
    auto amount_orig = amount;
    while(amount--) {
        int random = distribution(generator);
        std::string str = std::to_string(random);
        std::ofstream output;
        output.open(file_prefix + "_" + str + ".png");
        output << generate_png(str);
        output.close();
        disp.update((float) (amount_orig - amount) / amount_orig);
    }
    return 0;
}
