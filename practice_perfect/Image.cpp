#include "Image.h"

Image::Image(const std::string& file_path,
    const std::string& input_window_name,
    const std::string& output_window_name) {
    src_ = cv::imread(file_path.c_str());
    dst_.create(src_.size(), src_.type());

    input_window_name_ = input_window_name;
    output_window_name_ = output_window_name;
}

Image::Image(cv::Mat& src) {
    src.copyTo(src_);
    dst_.create(src_.size(), src_.type());
}

Image::~Image() {

}

cv::Mat& Image::GetSrcImage() {
    return src_;
}
cv::Mat& Image::GetDstImage() {
    return dst_;
}

void Image::ShowSrcImage() const {
    static bool first_run = true;
    if (first_run) {
        first_run = false;
        cv::namedWindow(input_window_name_, cv::WINDOW_AUTOSIZE);
    }
    cv::imshow(input_window_name_, src_);
}

void Image::ShowDstImage() const {
    static bool first_run = true;
    if (first_run) {
        first_run = false;
        cv::namedWindow(output_window_name_, cv::WINDOW_AUTOSIZE);
    }
    cv::imshow(output_window_name_, dst_);
}

bool Image::Empty() const {
    return src_.empty();
}