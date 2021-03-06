#ifndef GENERATOR_H
#define GENERATOR_H

#include <string>
#include <vector>

std::string generate_png(const std::string &text);
std::string generate_png_light(const std::string &text);
std::vector<float> generate_vec(const std::string &text);
std::vector<float> generate_vec_light(const std::string &text);
struct captcha {
    std::string png;
    std::vector<float> vec;
};
captcha generate(const std::string &text);

#endif
