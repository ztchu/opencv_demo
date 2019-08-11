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

void ImageProcessor::Dilate(Image& image, int pos) const {
    cv::Mat structure_element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * pos + 1, 2 * pos + 1));
    cv::dilate(image.GetSrcImage(), image.GetDstImage(), structure_element);
}

void ImageProcessor::Erode(Image& img, int pos) const {
    cv::Mat structure_element = cv::getStructuringElement(cv::MORPH_RECT,
        cv::Size(2 * pos + 1, 2 * pos + 1));
    cv::erode(img.GetSrcImage(), img.GetDstImage(), structure_element);
}

// MORPH_TOPHAT/MORPH_BLACKHAT/MORPH_OPEN/MORPH_CLOSE/MORPH_GRADIENT
void ImageProcessor::MorphologyOperation(Image& img, int operation) const {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(img.GetSrcImage(), img.GetDstImage(), cv::MORPH_BLACKHAT, kernel);
}

void ImageProcessor::ExtractHorizontalAndVeticalLine(Image& img, bool is_lookup_hline) const {
    // 1.Convert to gray image.
    cv::cvtColor(img.GetSrcImage(), img.GetDstImage(), cv::COLOR_BGR2GRAY);

    // 2.Convert to binary image, make sure the background is black.
    Image binary_image(img.GetDstImage());
    cv::adaptiveThreshold(~binary_image.GetSrcImage(), binary_image.GetDstImage(), 255,
        cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);

    if (is_lookup_hline) {
        // 3.Get horizontal structuring element.
        cv::Mat h_kernel = cv::getStructuringElement(cv::MORPH_RECT,
            cv::Size(binary_image.GetDstImage().cols / 16, 1));
        // 4.Erode
        //cv::erode(binary_image.GetDstImage(), binary_image.GetSrcImage(), h_kernel);
        // 5.Dilate
        //cv::dilate(binary_image.GetSrcImage(), img.GetDstImage(), h_kernel);

        // morphology open
        cv::morphologyEx(binary_image.GetDstImage(), img.GetDstImage(), cv::MORPH_OPEN, h_kernel);
    }
    else {
        // 3.Get vertical structuring element.
        cv::Mat v_kernel = cv::getStructuringElement(cv::MORPH_RECT,
            cv::Size(1, binary_image.GetDstImage().rows / 16));
        // 4.Erode
        //cv::erode(binary_image.GetDstImage(), binary_image.GetSrcImage(), v_kernel);
        // 5.Dilate
        //cv::dilate(binary_image.GetSrcImage(), img.GetDstImage(), v_kernel);
        
        // morphology open
        cv::morphologyEx(binary_image.GetDstImage(), img.GetDstImage(), cv::MORPH_OPEN, v_kernel);
    }

    cv::bitwise_not(img.GetDstImage(), img.GetDstImage());
}

void ImageProcessor::ExtractChars(Image& img) const {
    // 1.Convert to gray image.
    cv::cvtColor(img.GetSrcImage(), img.GetDstImage(), cv::COLOR_BGR2GRAY);

    // 2.Convert to binary image, make sure the background is black.
    Image binary_image(img.GetDstImage());
    cv::adaptiveThreshold(~binary_image.GetSrcImage(), binary_image.GetDstImage(), 255,
        cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 7, -2);

    // binary_image.ShowDstImage();

    // 3.Get rect structuring element.
    cv::Mat rect_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));

    // 4.Erode
    cv::erode(binary_image.GetDstImage(), binary_image.GetSrcImage(), rect_kernel);
    // 5.Dilate
    cv::dilate(binary_image.GetSrcImage(), img.GetDstImage(), rect_kernel);

    // morphology open
    //cv::morphologyEx(binary_image.GetDstImage(), img.GetDstImage(), cv::MORPH_OPEN, rect_kernel);

    cv::bitwise_not(img.GetDstImage(), img.GetDstImage());
    cv::blur(img.GetDstImage(), img.GetDstImage(), cv::Size(3, 3), cv::Point(-1, -1));
}

void ImageProcessor::PyramidUp(Image& img) const {
    cv::pyrUp(img.GetSrcImage(), img.GetDstImage(), 
        cv::Size(2 * img.GetSrcImage().cols, 2 * img.GetSrcImage().rows));
}

void ImageProcessor::PyramidDown(Image& img) const {
    cv::pyrDown(img.GetSrcImage(), img.GetDstImage(),
        cv::Size(img.GetSrcImage().cols / 2, img.GetSrcImage().rows / 2));
}

void ImageProcessor::Dog(Image& img) const {
    cv::cvtColor(img.GetSrcImage(), img.GetDstImage(), cv::COLOR_BGR2GRAY);

    cv::Mat first_blur;
    cv::GaussianBlur(img.GetDstImage(), first_blur, cv::Size(3, 3), 0, 0);
    
    cv::Mat second_blur;
    cv::GaussianBlur(first_blur, second_blur, cv::Size(3, 3), 0, 0);
    
    cv::subtract(first_blur, second_blur, img.GetDstImage());

    cv::normalize(img.GetDstImage(), img.GetDstImage(), 255, 0, cv::NORM_MINMAX);
}

void ImageProcessor::ThresholdOperation(Image& img, double threshold_value, double threshold_max, int op) const {
    cv::threshold(img.GetSrcImage(), img.GetDstImage(), threshold_value, threshold_max, op | cv::THRESH_OTSU);
}

void ImageProcessor::RobertKernelX(Image& img) const {
    cv::Mat kernel = (cv::Mat_<int>(2, 2) << 1, 0, 0, -1);
    cv::filter2D(img.GetSrcImage(), img.GetDstImage(), -1, kernel);
}

void ImageProcessor::RobertKernelY(Image& img) const {
    cv::Mat kernel = (cv::Mat_<int>(2, 2) << 0, 1, -1, 0);
    cv::filter2D(img.GetSrcImage(), img.GetDstImage(), -1, kernel);
}

void ImageProcessor::SobelKernelX(Image& img) const {
    cv::Mat kernel = (cv::Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::filter2D(img.GetSrcImage(), img.GetDstImage(), -1, kernel);
}

void ImageProcessor::SobelKernelY(Image& img) const {
    cv::Mat kernel = (cv::Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::filter2D(img.GetSrcImage(), img.GetDstImage(), -1, kernel);
}

void ImageProcessor::LaplaceKernel(Image& img) const {
    cv::Mat kernel = (cv::Mat_<int>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    cv::filter2D(img.GetSrcImage(), img.GetDstImage(), -1, kernel);
}

void ImageProcessor::SobelGradient(Image& img) const {
    // 1. gaussian blur
    cv::GaussianBlur(img.GetSrcImage(), img.GetDstImage(), cv::Size(3, 3), 0, 0);

    // 2.bgr to gray
    cv::Mat gray_img;
    cv::cvtColor(img.GetDstImage(), gray_img, cv::COLOR_BGR2GRAY);

    // 3.sobel x gradient
    cv::Mat xgrad;
    //cv::Sobel(gray_img, xgrad, CV_16S, 1, 0);
    cv::Scharr(gray_img, xgrad, CV_16S, 1, 0);
    cv::convertScaleAbs(xgrad, xgrad);    // saturate<uchar>)(|alpha*src+beta|)
    cv::imshow("xgrad", xgrad);

    // 4.sobel y gradient
    cv::Mat ygrad;
    //cv::Sobel(gray_img, ygrad, CV_16S, 0, 1);
    cv::Scharr(gray_img, ygrad, CV_16S, 1, 0);
    cv::convertScaleAbs(ygrad, ygrad);
    cv::imshow("ygrad", ygrad);

    // 5. add weighted
    cv::addWeighted(xgrad, 0.5, ygrad, 0.5, 0, img.GetDstImage());
}

void ImageProcessor::Laplacian(Image& img) const {
    // 1. gaussian blur
    cv::GaussianBlur(img.GetSrcImage(), img.GetDstImage(), cv::Size(3, 3), 0);

    // 2. convert color image to gray image.
    cv::Mat gray_image;
    cv::cvtColor(img.GetDstImage(), gray_image, cv::COLOR_BGR2GRAY);

    // 3.laplacian
    cv::Mat laplacian_image;
    cv::Laplacian(gray_image, laplacian_image, CV_16S, 3);

    // 4.convert scale abs
    cv::convertScaleAbs(laplacian_image, laplacian_image);

    // 5.threshold
    cv::threshold(laplacian_image, img.GetDstImage(), 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
}