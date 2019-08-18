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

    void Dilate(Image& img, int pos) const;
    void Erode(Image& img, int pos) const;

    void MorphologyOperation(Image& img, int operation) const;
    
    void ExtractHorizontalAndVeticalLine(Image& img, bool is_lookup_hline) const;
    void ExtractChars(Image& img) const;

    void PyramidUp(Image& img) const;
    void PyramidDown(Image& img) const;

    void Dog(Image& img) const;

    void ThresholdOperation(Image& img, double threshold_value,
        double threshold_max, int op) const;

    void RobertKernelX(Image& img) const;
    void RobertKernelY(Image& img) const;

    void SobelKernelX(Image& img) const;
    void SobelKernelY(Image& img) const;
    void LaplaceKernel(Image& img) const;

    void SobelGradient(Image& img) const;

    void Laplacian(Image& img) const;

    void CannyEdgeDetection(Image& img, int threshold_value,
        int high_threshold_value) const;

    void HoughLineDetection(Image& img) const;

    void HoughCircleDetection(Image& img) const;

    void CalculateHistogram(Image& img) const;

private:
    cv::Mat kernel_;
};

