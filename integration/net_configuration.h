#include <string>
#include <tiny_cnn/tiny_cnn.h>

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

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
    nn << convolutional_layer<tan_h>(220, 220, 5, 1, 6)  // C1, 1@220x220-in, 6@216x216-out
       << average_pooling_layer<tan_h>(216, 216, 6, 4)   // S2, 6@216x216-in, 6@54x54-out
       << convolutional_layer<tan_h>(54, 54, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@54x54-in, 16@50x50-out
       << average_pooling_layer<tan_h>(50, 50, 16, 2)  // S4, 16@50x50-in, 16@25x25-out
       << convolutional_layer<tan_h>(25, 25, 5, 16, 120) // C5, 16@25x25-in, 120@21x21-out
       << fully_connected_layer<tan_h>(52920, 6);       // F6, 52920-in, 6-out

    // nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
    //    << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
    //    << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
    //         connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
    //    << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
    //    << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
    //    << fully_connected_layer<tan_h>(120, 1);       // F6, 120-in, 1-out
}

vec_t round(const vec_t &s) {
    vec_t r = s;
    for(auto &f : r) f = round(f*10)/10;
    return r;
}

int restore_int(float a) {
    return (int) (((a + 1.0) / 2) * 10);
}

bool compare(vec_t a, vec_t b) {
    if(a.size() != b.size()) return false;
    for(size_t i = 0; i < a.size(); ++i)
        if(fabs(a[i] - b[i]) >= 0.01) return false;
    return true;
}

std::string print(vec_t v) {
    std::string res;
    for(auto &t : v) {
        if(!res.empty()) res += ' ';
        res += to_string(t);
    }
    return res;
}

vec_t code_to_vec(const std::string &code) {
    vec_t result;
    for(char s : code) if(s) result.push_back(0.2 * (float) (s - '0') - 1.0);
    return result;
}

void vector_rescale(vec_t &v) {
    // was 0..1, now -1..1
    for(auto &f : v) f = f*2 - 1.0;
}
