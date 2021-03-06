#include <iostream>
#include <string>
#include <random>
#include <sstream>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <assert.h>
#include <Magick++.h>

#include "generator.h"

#define stringify(s) #s
#define tostring(x) stringify(x)

static unsigned int MAX_DEPTH = std::numeric_limits<Magick::Quantum>::max();

static bool white(const Magick::PixelPacket *p) {
    return (MAX_DEPTH == p->red && MAX_DEPTH == p->blue && MAX_DEPTH == p->green);
}

static unsigned int left_pos(const Magick::Image &image) {
    const Magick::Geometry &g = image.size();
    int width = g.width(), height = g.height();
    const Magick::PixelPacket *packet = image.getConstPixels(0, 0, width, height);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            const Magick::PixelPacket *p = packet + y * width + x;
            if (!white(p)) return x;
        }
    }
    return 0;
}

static unsigned int right_pos(const Magick::Image &image) {
    const Magick::Geometry &g = image.size();
    int width = g.width(), height = g.height();
    const Magick::PixelPacket *packet = image.getConstPixels(0, 0, width, height);
    for (int x = width - 1; x >= 0; --x) {
        for (int y = 0; y < height; ++y) {
            const Magick::PixelPacket *p = packet + y * width + x;
            if (!white(p)) return x;
        }
    }
    return 0;
}

static std::vector<std::string> get_ls(const std::string &dir) {
    DIR *dirp = opendir(dir.c_str());
    struct dirent *dp;
    std::vector<std::string> result;
    while((dp = readdir(dirp)) != NULL)
        if(dp->d_name[0] == '.') continue;
        else result.emplace_back(dp->d_name);
    (void)closedir(dirp);
    return result;
}

static std::string get_random_font() {
    std::stringstream stream;
    static std::vector<std::string> fonts = get_ls(tostring(FONT_PATH));
    stream << "@" << tostring(FONT_PATH) << "/" << fonts.at(random() % fonts.size());
    return stream.str().c_str();
}

static void print_digit(Magick::Image &image, char digit, bool first) {
    Magick::Image tmp(Magick::Geometry(IMAGE_HEIGHT, IMAGE_HEIGHT), Magick::Color("white"));
    tmp.antiAlias(true);
    tmp.font(get_random_font());
    tmp.fontPointsize(50 + (random() % 3 - 1));
    tmp.fillColor(Magick::Color("white"));
    tmp.strokeColor(Magick::Color("black"));
    char one_digit[2];
    one_digit[0]=digit;
    one_digit[1]=0;
    tmp.annotate(one_digit,Magick::Geometry(IMAGE_HEIGHT, IMAGE_HEIGHT), Magick::NorthGravity, static_cast<double>(random() % 20 - 10));
    if (1 == random() % 2) {
        tmp.gaussianBlur(2, 2);
    }
    tmp.transparent(Magick::Color("white"));
    long pos = (first ? 0 : right_pos(image) - left_pos(tmp) + 4);
    image.composite(tmp, pos, -8, Magick::OverCompositeOp);
}

static void print_lines(Magick::Image &image) {
    typedef std::mt19937 Gen;
    typedef std::normal_distribution<double> Dist;
    Gen r(time(NULL));
    Dist d(100, 15);
    auto rr = std::bind(d, r);
    image.fillColor(Magick::Color("white"));
    image.strokeColor(Magick::Color("gray"));
    const Magick::Geometry &g = image.size();
    int count = (static_cast<int>(rr()) / 40), dist = g.height() / count;
    for (long i = 0; i < count; ++i) {
        Magick::DrawableLine line(0, i * dist+10, g.width() - 1, i * dist);
        image.draw(line);
    }
    image.gaussianBlur(1, 2);
}

static void setup_image(Magick::Image &image) {
    image.antiAlias(false);
    image.interlaceType(Magick::LineInterlace);
    image.type(Magick::TrueColorType);
    image.matte(true);
    image.matteColor(Magick::Color("white"));
    image.borderColor(Magick::Color("white"));
}

static void print_image(Magick::Image &image, const std::string &number) {
    for(std::string::size_type i=0;i<number.size();i++)
        print_digit(image,number[i],!i);
    double sw = random() % 10 - 20;
    image.swirl(sw);
    image.wave(8, image.size().width() / (random() % 3 + 1));
    image.trim();
    long nu_width = right_pos(image) - left_pos(image) + 4;
    Magick::Image centered(Magick::Geometry(IMAGE_WIDTH, IMAGE_HEIGHT), Magick::Color("white"));
    centered.composite(image, (IMAGE_WIDTH-nu_width)/2, 0, Magick::OverCompositeOp);
    image=centered;
    Magick::Image tmp(Magick::Geometry(IMAGE_WIDTH, IMAGE_HEIGHT), Magick::Color("white"));
    print_lines(tmp);
    tmp.wave(8, image.size().width() / (random() % 5 + 2));
    tmp.trim();
    image.composite(tmp, 0, 0, Magick::MultiplyCompositeOp);
    image.transparent(Magick::Color("white"));
}

static void print_image_light(Magick::Image &image, const std::string &number) {
    for(std::string::size_type i=0;i<number.size();i++)
        print_digit(image,number[i],!i);
    double sw = random() % 10 - 20;
    image.swirl(sw);
    image.wave(8, image.size().width() / (random() % 3 + 1));
    image.trim();
    long nu_width = right_pos(image) - left_pos(image) + 4;
    Magick::Image centered(Magick::Geometry(IMAGE_HEIGHT, IMAGE_HEIGHT), Magick::Color("white"));
    centered.composite(image, (IMAGE_HEIGHT-nu_width)/2, 0, Magick::OverCompositeOp);
    image=centered;
    Magick::Image tmp(Magick::Geometry(IMAGE_HEIGHT, IMAGE_HEIGHT), Magick::Color("white"));
    print_lines(tmp);
    tmp.wave(8, image.size().width() / (random() % 5 + 2));
    tmp.trim();
    image.composite(tmp, 0, 0, Magick::MultiplyCompositeOp);
    image.transparent(Magick::Color("white"));
}
std::string generate_png(const std::string &text) {
    Magick::Image image(Magick::Geometry(IMAGE_WIDTH, IMAGE_HEIGHT), Magick::Color("white"));
    setup_image(image);
    print_image(image, text);
    image.type(Magick::GrayscaleType);
    Magick::Blob blob;
    image.magick("PNG");
    image.write(&blob);
    return std::string((const char*) blob.data(), blob.length());
}

std::string generate_png_light(const std::string &text) {
    Magick::Image image(Magick::Geometry(IMAGE_HEIGHT, IMAGE_HEIGHT), Magick::Color("white"));
    setup_image(image);
    print_image_light(image, text);
    image.type(Magick::GrayscaleType);
    Magick::Blob blob;
    image.magick("PNG");
    image.write(&blob);
    return std::string((const char*) blob.data(), blob.length());
}

// static size_t make_square_pre(std::vector<float_t> &v) {
//     if(IMAGE_WIDTH == IMAGE_HEIGHT) return 0;
//     assert(IMAGE_WIDTH > IMAGE_HEIGHT);
//     size_t addon = IMAGE_WIDTH * ((IMAGE_WIDTH - IMAGE_HEIGHT) / 2);
//     for(size_t i = 0; i < addon; ++i) v[i] = 1.0;
//     return addon;
// }
// 
// static void make_square_post(std::vector<float_t> &v) {
//     if(IMAGE_WIDTH == IMAGE_HEIGHT) return;
//     assert(IMAGE_WIDTH > IMAGE_HEIGHT);
//     size_t addon = IMAGE_WIDTH * (IMAGE_WIDTH - IMAGE_HEIGHT - ((IMAGE_WIDTH - IMAGE_HEIGHT) / 2));
//     size_t offset = IMAGE_WIDTH * (IMAGE_WIDTH - (addon/IMAGE_WIDTH));
//     for(size_t i = offset; i < offset + addon; ++i) v[i] = 1.0;
// }

std::vector<float_t> generate_vec(const std::string &text) {
    Magick::Image image(Magick::Geometry(IMAGE_WIDTH, IMAGE_HEIGHT), Magick::Color("white"));
    setup_image(image);
    print_image(image, text);
    image.type(Magick::GrayscaleType);
    std::vector<float_t> result;
    result.resize(IMAGE_WIDTH * IMAGE_HEIGHT);
    size_t offs = 0; // make_square_pre(result);
    image.write(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, "R", Magick::FloatPixel, &(result[offs]));
    // make_square_post(result);
    return result;
}

std::vector<float_t> generate_vec_light(const std::string &text) {
    Magick::Image image(Magick::Geometry(IMAGE_HEIGHT, IMAGE_HEIGHT), Magick::Color("white"));
    setup_image(image);
    print_image_light(image, text);
    image.type(Magick::GrayscaleType);
    std::vector<float_t> result;
    result.resize(IMAGE_HEIGHT * IMAGE_HEIGHT);
    size_t offs = 0; // make_square_pre(result);
    image.write(0, 0, IMAGE_HEIGHT, IMAGE_HEIGHT, "R", Magick::FloatPixel, &(result[offs]));
    // make_square_post(result);
    return result;
}

captcha generate(const std::string &text) {
    Magick::Image image(Magick::Geometry(IMAGE_WIDTH, IMAGE_HEIGHT), Magick::Color("white"));
    setup_image(image);
    print_image(image, text);
    image.type(Magick::GrayscaleType);
    captcha result;
    result.vec.resize(IMAGE_WIDTH * IMAGE_HEIGHT);
    size_t offs = 0; // make_square_pre(result.vec);
    image.write(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, "R", Magick::FloatPixel, &(result.vec[offs]));
    // make_square_post(result.vec);
    Magick::Blob blob;
    image.magick("PNG");
    image.write(&blob);
    result.png.assign((const char*) blob.data(), blob.length());
    return result;
}
