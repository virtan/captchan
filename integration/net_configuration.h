#include <string>
#include <tiny_dnn/tiny_dnn.h>

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

void construct_net(network<graph>& nn) {
    auto conv1 = new convolutional_layer<tan_h>(220, 60, 5, 1, 32);
    auto maxpool1 = new average_pooling_layer<tan_h>(216, 56, 32, 2);
    auto conv2 = new convolutional_layer<tan_h>(108, 28, 5, 32, 32);
    auto maxpool2 = new average_pooling_layer<tan_h>(104, 24, 32, 2);

    *conv1 << *maxpool1 << *conv2 << *maxpool2;

    std::vector<layer*> outputs;
    for(int i = 0; i < 1; ++i) {
        auto drop = new dropout_layer(19968, 0.5);
        auto dense = new fully_connected_layer<tan_h>(19968, 1);
        *maxpool2 << *drop << *dense;
        outputs.push_back(dense);
    }

    construct_graph(nn, {conv1}, outputs);
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
        if(std::isnan(a[i]) || std::isnan(b[i]) || fabs(a[i] - b[i]) >= 0.01) return false;
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
