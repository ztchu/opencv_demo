#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class Image
{
public:
    Image(const std::string& file_path,
        const std::string& input_window_name = "input image",
        const std::string& output_window_name = "output image");
    Image(cv::Mat& src,
        const std::string& input_window_name = "input image",
        const std::string& output_window_name = "output image");
    ~Image();
    cv::Mat& GetSrcImage();
    cv::Mat& GetDstImage();

    // If is_original is true, show the original image,
    // else show bitwise_not(image).
    void ShowSrcImage(bool is_original = true) const;
    void ShowDstImage(bool is_original = true) const;

    bool Empty() const;
    std::string GetOutputWindowName() const;

private:
    std::string input_window_name_;
    std::string output_window_name_;
    cv::Mat src_;
    cv::Mat dst_;
};

