#include "ImageProcessor.h"

#include "logger.h"

void ImageProcessor::SetKernel(const cv::Mat& kernel) {
    kernel_ = kernel;
}

void ImageProcessor::MaskOperation(Image& img) const {
    cv::Mat& src = img.GetSrcImage();

    int offset = src.channels();
    int cols = src.cols * offset;
    int rows = src.rows;

    for (int row = 1; row < rows - 1; ++row) {
        const uchar* prev = src.ptr<uchar>(row - 1);
        const uchar* cur = src.ptr<uchar>(row);
        const uchar* next = src.ptr<uchar>(row + 1);
        uchar* dst = img.GetDstImage().ptr<uchar>(row);
        for (int col = offset; col < cols - offset; ++col) {
            dst[col] = cv::saturate_cast<uchar>(5 * cur[col] - cur[col - offset] - cur[col + offset] - prev[col] - next[col]);
        }
    }
}

void ImageProcessor::Filter2DMask(Image& img) const {
    if (kernel_.empty()) {
        LOG_WARN << "The kernel is empty.";
        return;
    }
    cv::filter2D(img.GetSrcImage(), img.GetDstImage(), -1, kernel_);
}

void ImageProcessor::BitwiseNot(Image& img) const {
    cv::Mat& src = img.GetSrcImage();
    cv::Mat& dst = img.GetDstImage();
    for (size_t row = 0; row < src.rows; ++row) {
        for (size_t col = 0; col < src.cols; ++col) {
            if (src.channels() == 1) {
                dst.at<uchar>(row, col) = 255 - src.at<uchar>(row, col);
            }
            else if (src.channels() == 3) {
                dst.at<cv::Vec3b>(row, col)[0] = 255 - src.at<cv::Vec3b>(row, col)[0];
                dst.at<cv::Vec3b>(row, col)[1] = 255 - src.at<cv::Vec3b>(row, col)[1];
                dst.at<cv::Vec3b>(row, col)[2] = 255 - src.at<cv::Vec3b>(row, col)[2];
            }
        }
    }
}

void ImageProcessor::Blur(Image& img) const {
    cv::blur(img.GetSrcImage(), img.GetDstImage(), cv::Size(3, 3));
}

void ImageProcessor::GaussianBlur(Image& img) const {
    cv::GaussianBlur(img.GetSrcImage(), img.GetDstImage(), cv::Size(5, 5), 11, 11);
}

void ImageProcessor::MediaBlur(Image& img) const {
    cv::medianBlur(img.GetSrcImage(), img.GetDstImage(), 7);
}

void ImageProcessor::BilateralFilter(Image& img) const {
    cv::bilateralFilter(img.GetSrcImage(), img.GetDstImage(), 15, 100, 3);
}

cv::Mat ImageProcessor::GenerateGaussianTemplate(int size, double sigma) const {
    int real_size = size;
    if (real_size % 2 == 0) {
        real_size += 1;
    }

    cv::Mat result(cv::Size(real_size, real_size), CV_32FC1);
    const float pre_const = 1 / (2 * CV_PI * sigma * sigma);

    const float coefficient = -1 / (2 * sigma * sigma);
    const int center = real_size / 2;
    for (int row = 0; row < real_size; ++row) {
        float square_x = pow(row - center, 2);
        for (int col = 0; col < real_size; ++col) {
            float square_y = pow(col - center, 2);
            result.at<float>(row, col) = pre_const * exp(coefficient * (square_x + square_y));
        }
    }

    float scale = 1 / result.at<float>(0, 0);
    for (int row = 0; row < real_size; ++row) {
        for (int col = 0; col < real_size; ++col) {
            result.at<float>(row, col) *= scale;
        }
    }
    return result;
}
