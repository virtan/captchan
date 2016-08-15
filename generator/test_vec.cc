#include <iostream>
#include <fstream>
#include <unistd.h>
#include <stdlib.h>

#include "generator.h"

int main(int argc, char **argv) {
    srandom(time(NULL));
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <string>\n";
        exit(1);
    }
    std::ofstream output;
    output.open("generated.txt");
    for(float a : generate_vec(argv[1]))
        output << a << ' ';
    output.close();
    return 0;
}
