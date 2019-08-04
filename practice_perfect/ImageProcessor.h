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

    // Implements of cv::bitwise_not()
    void BitwiseNot(Image& img) const;

    // blur
    void Blur(Image& img) const;
    void GaussianBlur(Image& img) const;
    void MediaBlur(Image& img) const;
    void BilateralFilter(Image& img) const;

    // reference: https://blog.csdn.net/fzhykx/article/details/79532864
    cv::Mat GenerateGaussianTemplate(int size, double sigma) const;

private:
    cv::Mat kernel_;
};

