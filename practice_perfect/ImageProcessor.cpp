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