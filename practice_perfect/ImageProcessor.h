#pragma once
#include <opencv2/opencv.hpp>

#include "Image.h"

class ImageProcessor
{
public:
    ImageProcessor() {}

    // If you want to use new kernel in your program,
    // you should call SetKernel() before any other operation.
    void SetKernel(const cv::Mat& kernel);
    void MaskOperation(Image & img) const;
    void Filter2DMask(Image& img) const;

private:
    cv::Mat kernel_;
};

