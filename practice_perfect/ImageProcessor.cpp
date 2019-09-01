#include "ImageProcessor.h"

#include "logger.h"

#include <vector>

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
            dst[col] = cv::saturate_cast<uchar>(5 * cur[col] - cur[col - offset]
                - cur[col + offset] - prev[col] - next[col]);
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

void ImageProcessor::ThresholdOperation(Image& img, double threshold_value,
    double threshold_max, int op) const {
    cv::threshold(img.GetSrcImage(), img.GetDstImage(), threshold_value,
        threshold_max, op | cv::THRESH_OTSU);
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

void ImageProcessor::CannyEdgeDetection(Image& img, int low_threshold_value, int high_threshold_value) const {
    // 1.gaussian blur
    cv::GaussianBlur(img.GetSrcImage(), img.GetDstImage(), cv::Size(3, 3), 0);

    // 2.convert color image to gray
    cv::Mat gray_image;
    cv::cvtColor(img.GetDstImage(), gray_image, cv::COLOR_BGR2GRAY);

    // 3.canny edge detection
    cv::Mat canny_edge;
    cv::Canny(gray_image, canny_edge, low_threshold_value, high_threshold_value);

    // 4.copy src by mask of edge
    img.GetDstImage() = cv::Mat::zeros(img.GetSrcImage().size(), img.GetSrcImage().type());
    img.GetSrcImage().copyTo(img.GetDstImage(), canny_edge);
}

void ImageProcessor::HoughLineDetection(Image& img) const {
    // 1. canny
    cv::Canny(img.GetSrcImage(), img.GetDstImage(), 100, 200);

    // 2.hough line 
    std::vector<cv::Vec4f> lines;
    cv::HoughLinesP(img.GetDstImage(), lines, 1, CV_PI / 180.0, 10, 50, 5);

    // 3.convert to color, for draw color line.
    cv::cvtColor(img.GetDstImage(), img.GetDstImage(), cv::COLOR_GRAY2BGR);

    //img.GetDstImage() = cv::Mat::zeros(img.GetDstImage().size(), img.GetDstImage().type());
    cv::Scalar color(0, 0, 255);
    std::cout << lines.size() << std::endl;
    for (auto i = 0; i < lines.size(); ++i) {
        cv::Vec4f line = lines[i];
        cv::line(img.GetDstImage(), cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), color, 2);
    }
}

void ImageProcessor::HoughCircleDetection(Image& img) const {
    // 1.median blur
    cv::medianBlur(img.GetSrcImage(), img.GetDstImage(), 3);

    // 2.convert color image to gray
    cv::Mat gray_image;
    cv::cvtColor(img.GetDstImage(), gray_image, cv::COLOR_BGR2GRAY);

    // 3.hough circle detection.
    std::vector<cv::Vec3f> possible_circle;
    cv::HoughCircles(gray_image, possible_circle, cv::HOUGH_GRADIENT, 1, 10, 100, 30, 5, 100);

    // 4.draw to output image
    for (auto i = 0; i < possible_circle.size(); ++i) {
        auto cc = possible_circle[i];
        cv::circle(img.GetDstImage(), cv::Point(cc[0], cc[1]), cc[2], cv::Scalar(0, 0, 255), 2);
    }
}

void ImageProcessor::CalculateHistogram(Image& img) const {
    // 1. split bgr
    std::vector<cv::Mat> bgr_planes;
    cv::split(img.GetSrcImage(), bgr_planes);
    
    // 2. calculate hist
    int hist_size = 256;
    float ranges[] = { 0, 256 };
    const float* hist_ranges = ranges;
    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &hist_size, &hist_ranges, true, false);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &hist_size, &hist_ranges, true, false);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &hist_size, &hist_ranges, true, false);

    // 3. normalize
    int hist_height = 256;
    cv::normalize(b_hist, b_hist, 0, hist_height, cv::NORM_MINMAX);
    cv::normalize(g_hist, g_hist, 0, hist_height, cv::NORM_MINMAX);
    cv::normalize(r_hist, r_hist, 0, hist_height, cv::NORM_MINMAX);
    
    // 4. draw hist
    int hist_width = 512;
    int bin_width = hist_width / hist_size;
    img.GetDstImage() = cv::Mat(hist_height, hist_width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (size_t i = 1; i < hist_size; ++i) {
        cv::line(img.GetDstImage(), cv::Point((i - 1) * bin_width, hist_height - cvRound(b_hist.at<float>(i - 1))),
            cv::Point(i * bin_width, hist_height - cvRound(b_hist.at<float>(i))), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        cv::line(img.GetDstImage(), cv::Point((i - 1) * bin_width, hist_height - cvRound(g_hist.at<float>(i - 1))),
            cv::Point(i * bin_width, hist_height - cvRound(g_hist.at<float>(i))), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::line(img.GetDstImage(), cv::Point((i - 1) * bin_width, hist_height - cvRound(r_hist.at<float>(i - 1))),
            cv::Point(i * bin_width, hist_height - cvRound(r_hist.at<float>(i))), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }
}

double ImageProcessor::CompareHistogram(Image& img_lhs, Image& img_rhs, int comp_method) const {
    // 1.Convert bgr to hsv.
    cv::cvtColor(img_lhs.GetSrcImage(), img_lhs.GetDstImage(), cv::COLOR_BGR2HSV);
    cv::cvtColor(img_rhs.GetSrcImage(), img_rhs.GetDstImage(), cv::COLOR_BGR2HSV);

    // 2.Calculate histogram.
    cv::Mat img_lhs_hist, img_rhs_hist;
    int channels[] = { 0, 1 };
    int h_size = 50;
    int s_size = 60;
    int hs_size[] = { h_size, s_size };
    float h_ranges[] = { 0, 180 };
    float s_rangs[] = { 0, 256 };
    const float* ranges[] = { h_ranges , s_rangs };
    cv::calcHist(&img_lhs.GetDstImage(), 1, channels, cv::Mat(), img_lhs_hist,
        2, hs_size, ranges, true, false);
    cv::calcHist(&img_rhs.GetDstImage(), 1, channels, cv::Mat(), img_rhs_hist,
        2, hs_size, ranges, true, false);

    // 3.Normalize
    cv::normalize(img_lhs_hist, img_lhs_hist, 0, 1, cv::NORM_MINMAX);
    cv::normalize(img_rhs_hist, img_rhs_hist, 0, 1, cv::NORM_MINMAX);
    
    // 4.Compare
    return cv::compareHist(img_lhs_hist, img_rhs_hist, comp_method);
}

void ImageProcessor::BackProjectHist(Image& img) const {
    // 1.Convert bgr to hsv
    cv::cvtColor(img.GetSrcImage(), img.GetDstImage(), cv::COLOR_BGR2HSV);

    // 2.Extract hue component
    cv::Mat hue(img.GetDstImage().size(), img.GetDstImage().depth());
    const int fromto[] = { 0, 0 };
    cv::mixChannels(&img.GetDstImage(), 1, &hue, 1, fromto, 1);

    // 3.Calculate hist
    float ranges[] = { 0, 180 };
    const float* hist_ranges = { ranges };
    cv::Mat hue_hist;
    int hist_size = 12;
    cv::calcHist(&hue, 1, 0, cv::Mat(), hue_hist, 1, &hist_size, &hist_ranges, true, false);

    // 4. normalize
    cv::normalize(hue_hist, hue_hist, 0, 255, cv::NORM_MINMAX);

    // 5.Calculate histogram back projection.
    cv::Mat back_proj;
    cv::calcBackProject(&hue, 1, 0, hue_hist, back_proj, &hist_ranges);
    img.GetDstImage() = back_proj;
}

void ImageProcessor::TemplateMatch(Image& src_img, const cv::Mat& template_img, int method) const {
    // 1.Template match
    cv::Mat result(src_img.GetSrcImage().cols - template_img.cols + 1,
        src_img.GetSrcImage().rows - template_img.rows + 1, CV_32F);
    cv::matchTemplate(src_img.GetSrcImage(), template_img, result, method);

    // 2. normalize
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);

    // 3.Get min max loc
    cv::Point min_loc, max_loc;
    double min_value, max_value;
    cv::minMaxLoc(result, &min_value , &max_value , &min_loc, &max_loc);

    // 4.Draw rectangle.
    cv::Point temp_loc;
    if (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED) {
        temp_loc = min_loc;
    }
    else {
        temp_loc = max_loc;
    }

    cv::rectangle(src_img.GetSrcImage(), cv::Rect(temp_loc.x, temp_loc.y,
        template_img.cols, template_img.rows), cv::Scalar(255, 0, 0));
    src_img.GetDstImage() = result;
    cv::rectangle(src_img.GetDstImage(), cv::Rect(temp_loc.x, temp_loc.y,
        template_img.cols, template_img.rows), cv::Scalar(0, 255, 0));
}

void ImageProcessor::DiscoverContours(Image& img, int low_threshold_value) const {
    // 1. Convert color image to gray image.
    cv::cvtColor(img.GetSrcImage(), img.GetDstImage(), cv::COLOR_BGR2GRAY);

    // 2. Canny edge detection.
    cv::Mat edges;
    cv::Canny(img.GetDstImage(), edges, low_threshold_value, 2 * low_threshold_value, 3, false);

    // 3. Find contours.
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 4. Draw Contours.
    img.GetDstImage() = cv::Mat::zeros(img.GetSrcImage().size(), CV_8UC3);
    cv::RNG rng;
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::drawContours(img.GetDstImage(), contours, i, color, 1, 8, hierarchy, 0);
    }
}

void ImageProcessor::FindConvexHull(Image& img) const {
    // 1. Convert bgr to gray.
    cv::cvtColor(img.GetSrcImage(), img.GetDstImage(), cv::COLOR_BGR2GRAY);

    // 2. Image filtering.
    cv::Mat blur_image;
    cv::blur(img.GetDstImage(), blur_image, cv::Size(3, 3));

    // 3. Image binaryzation.
    cv::Mat bin_image;
    cv::threshold(blur_image, bin_image, 127, 255, cv::BORDER_DEFAULT);

    // 4. Find contours.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 5. Find convex hull.
    std::vector<std::vector<cv::Point>> convexs(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::convexHull(contours[i], convexs[i]);
    }

    // 6. Draw convex hull.
    cv::RNG rng;
    img.GetDstImage() = cv::Mat::zeros(img.GetSrcImage().size(), CV_8UC3);
    for (size_t i = 0; i < convexs.size(); ++i) {
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::drawContours(img.GetDstImage(), contours, i, color);
        cv::drawContours(img.GetDstImage(), convexs, i, color);
    }
}

void ImageProcessor::FindTarget(Image& img) const {
    // 1. Convert color image to gray.
    cv::cvtColor(img.GetSrcImage(), img.GetDstImage(), cv::COLOR_BGR2GRAY);

    // 2. Image filtering.
    cv::Mat filter_image;
    cv::blur(img.GetDstImage(), filter_image, cv::Size(3, 3));

    // 3. Image binaryzation.
    cv::Mat bin_image;
    cv::threshold(img.GetDstImage(), bin_image, 164, 255, cv::THRESH_BINARY);

    // 4. Find contours.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bin_image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 5. Find the smallest shape.
    std::vector<std::vector<cv::Point>> contours_ploy(contours.size());
    std::vector<cv::Rect> rects(contours.size());
    std::vector<cv::Point2f> circle_centers(contours.size());
    std::vector<float> circle_radius(contours.size());
    std::vector<cv::RotatedRect> min_rects(contours.size());
    std::vector<cv::RotatedRect> min_ellipse(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::approxPolyDP(contours[i], contours_ploy[i], 5, true);
        rects[i] = cv::boundingRect(contours_ploy[i]);
        cv::minEnclosingCircle(contours_ploy[i], circle_centers[i], circle_radius[i]);
        if (contours_ploy[i].size() > 4) {
            min_ellipse[i] = cv::fitEllipse(contours_ploy[i]);
            min_rects[i] = cv::minAreaRect(contours_ploy[i]);
        }
    }

    // 6. Draw shape.
    cv::RNG rng;
    cv::Point2f rect_points[4];
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        //cv::rectangle(img.GetSrcImage(), rects[i], color);
        //cv::circle(img.GetSrcImage(), circle_centers[i], circle_radius[i], color);
        if (contours[i].size() > 4) {
            cv::ellipse(img.GetSrcImage(), min_ellipse[i], color);
            min_rects[i].points(rect_points);
            for (size_t j = 0; j < 4; ++j) {
                cv::line(img.GetSrcImage(), rect_points[j], rect_points[(j + 1) % 4], color);
            }
        }
    }
}

void ImageProcessor::CalculateMoments(Image& img, int threshold_value) const {
    // 1.Convert color image to gray.
    cv::Mat gray_image;
    cv::cvtColor(img.GetSrcImage(), gray_image, cv::COLOR_BGR2GRAY);

    // 2.Gaussian blur.
    cv::Mat blur_image;
    cv::GaussianBlur(gray_image, blur_image, cv::Size(3, 3), 0);

    // 3.Canny edge detection.
    cv::Mat canny_image;
    cv::Canny(blur_image, canny_image, threshold_value, threshold_value * 2, 3, false);

    // 4.Find contours.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(canny_image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 5.Calculate moments.
    std::vector<cv::Moments> contours_moments(contours.size());
    std::vector<cv::Point> circle_centers(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        contours_moments[i] = cv::moments(contours[i]);
        circle_centers[i] = cv::Point(static_cast<float>(contours_moments[i].m10 / contours_moments[i].m00),
            static_cast<float>(contours_moments[i].m01 / contours_moments[i].m00));
    }

    // 6.Draw image.
    cv::RNG rng;
    img.GetDstImage() = cv::Mat::zeros(img.GetSrcImage().size(), CV_8UC3);
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::drawContours(img.GetDstImage(), contours, i, color);
        cv::circle(img.GetDstImage(), circle_centers[i], 3, color);
    }
}

void ImageProcessor::PointPolygonTest(Image& img, const cv::Point2f& test_point) const {
    // 1.Convert color image to gray.
    cv::Mat gray_image;
    cv::cvtColor(img.GetSrcImage(), gray_image, cv::COLOR_BGR2GRAY);

    // 2.Image filtering .
    cv::Mat filter_image;
    cv::blur(gray_image, filter_image, cv::Size(3, 3));

    // 3.Image binaryzation.
    cv::threshold(filter_image, img.GetDstImage(), 167, 255, cv::THRESH_OTSU);

    // 4.Find contours.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img.GetDstImage(), contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 5.Point polygon test.
    img.GetDstImage() = cv::Mat::zeros(img.GetSrcImage().size(), CV_8UC3);
    cv::RNG rng;
    cv::circle(img.GetDstImage(), test_point, 3, cv::Scalar(0, 0, 255));
    for (size_t i = 0; i < contours.size(); ++i) {
        // When flag is false, 1 represents inside the polygon, 0 represents on the edge.
        if (cv::pointPolygonTest(contours[i], test_point, false) >= 0) {
            cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            cv::drawContours(img.GetDstImage(), contours, i, color, 2);
        }
    }
}

void BinaryThresholdTrackbarCallback(int pos, void* user_data) {
    if (user_data == nullptr) {
        return;
    }
    Image& image = std::ref(*(static_cast<Image*>(user_data)));

    cv::threshold(image.GetSrcImage(), image.GetDstImage(), pos, 255, cv::THRESH_BINARY);

    image.ShowDstImage();
}

void ImageProcessor::FindBestBinayThreshold(Image& img) const {
    // 1. Convert color image to gray
    cv::Mat gray_image;
    cv::cvtColor(img.GetSrcImage(), gray_image, cv::COLOR_BGR2GRAY);

    // 2. Store src image.
    cv::Mat src_image = img.GetSrcImage();
    img.ShowSrcImage();

    // 2. Gaussian blur.
    cv::GaussianBlur(gray_image, img.GetSrcImage(), cv::Size(3, 3), 0);

    // 3. Draw trackbar.
    int thresh_value = 78;
    cv::createTrackbar("threshold value", img.GetOutputWindowName(), &thresh_value, 255,
        BinaryThresholdTrackbarCallback, &img);
    BinaryThresholdTrackbarCallback(thresh_value, &img);
}

void ImageProcessor::ImageSegmentation(Image& img) const {
    // 1. Change background to black.
    cv::Mat black_image;
    img.GetSrcImage().copyTo(black_image);
    for (size_t i = 0; i < black_image.rows; ++i) {
        for (size_t j = 0; j < black_image.cols; ++j) {
            if (black_image.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 255, 255)) {
                black_image.at<cv::Vec3b>(i, j)[0] = 0;
                black_image.at<cv::Vec3b>(i, j)[1] = 0;
                black_image.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }
    // cv::imshow("background image", black_image);
    
    // 2. Improve image contrast.
    cv::Mat laplace_kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
    cv::Mat contrast_image;
    cv::filter2D(black_image, contrast_image, CV_32F, laplace_kernel);
    //cv::imshow("filter2D image", contrast_image);
    cv::Mat float_image;
    black_image.convertTo(float_image, CV_32F);
    cv::Mat diff_image = float_image - contrast_image;
    diff_image.convertTo(diff_image, CV_8UC3);
    //cv::imshow("diff image", diff_image);

    // 3. Image binaryzation.
    cv::Mat gray_image;
    cv::cvtColor(diff_image, gray_image, cv::COLOR_BGR2GRAY);
    cv::Mat binary_image;
    cv::threshold(gray_image, binary_image, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    // cv::imshow("binary image", binary_image);

    // 4. Distance transform.
    cv::Mat dist_image;
    cv::distanceTransform(binary_image, dist_image, cv::DIST_L1, 3);
    
    // 5. Normalize the result of distance transform.
    cv::normalize(dist_image, dist_image, 0, 1, cv::NORM_MINMAX);
    // cv::imshow("distance image", dist_image);

    // 6. Redo image binaryzation.
    cv::threshold(dist_image, dist_image, 0.4, 1, cv::THRESH_BINARY);
    cv::imshow("binary image", dist_image);

    // 7. Erode 
    cv::Mat erode_kernel = cv::Mat::zeros(cv::Size(3, 3), CV_8UC1);
    cv::erode(dist_image, dist_image, erode_kernel);
    cv::imshow("erode image", dist_image);

    // 8. Find contours
    dist_image.convertTo(dist_image, CV_8U);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dist_image, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // 9. Draw contours
    cv::Mat markers = cv::Mat::zeros(dist_image.size(), CV_32SC1);
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::drawContours(markers, contours, i, cv::Scalar::all(i + 1), -1);
    }
    cv::circle(markers, cv::Point(5, 5), 3, cv::Scalar(255, 255, 255));

    // 10. water shed
    cv::watershed(diff_image, markers);
    cv::Mat mark = cv::Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    cv::bitwise_not(mark, mark);
    // cv::imshow("mark", mark);

    // 11. Coloring
    std::vector<cv::Vec3b> colors;
    for (size_t i = 0; i < contours.size(); ++i) {
        colors.emplace_back(cv::Vec3b(cv::theRNG().uniform(0, 255),
            cv::theRNG().uniform(0, 255), cv::theRNG().uniform(0, 255)));
    }

    img.GetDstImage() = cv::Mat::zeros(markers.size(), CV_8UC3);
    for (size_t i = 0; i < markers.rows; ++i) {
        for (size_t j = 0; j < markers.cols; ++j) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= contours.size()) {
                img.GetDstImage().at<cv::Vec3b>(i, j) = colors[index - 1];
            }
            else {
                img.GetDstImage().at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    img.ShowDstImage();
}