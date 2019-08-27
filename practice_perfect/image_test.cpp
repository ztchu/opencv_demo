#include "image_test.h"

#include <vector>

namespace image_test {

void TestMaskOperation() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return ;
    }
    lena.ShowSrcImage();

    ImageProcessor processor;
    processor.MaskOperation(lena);
    lena.ShowDstImage();
    cv::waitKey(0);
}

void TestFilter2D() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }
    lena.ShowSrcImage();

    ImageProcessor processor;
    cv::Mat kernel = (cv::Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    processor.SetKernel(kernel);
    processor.Filter2DMask(lena);

    lena.ShowDstImage();
    cv::waitKey(0);
}

void DilateTrackbarCallback(int pos, void* userdata) {
    if (userdata == nullptr) {
        return;
    }
    Image& image = std::ref(*(static_cast<Image*>(userdata)));
    ImageProcessor processor;
    processor.Dilate(image, pos);
    image.ShowDstImage();
}

void TestDilate() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return ;
    }

    int init_value = 3;
    const int max_value = 21;
    cv::createTrackbar("morphorlogy operation", lena.GetOutputWindowName(),
        &init_value, max_value, DilateTrackbarCallback, &lena);
    DilateTrackbarCallback(init_value, &lena);
    cv::waitKey(0);
}

void TestMorphology() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    int operation = cv::MORPH_BLACKHAT;
    processor.MorphologyOperation(lena, operation);
    lena.ShowDstImage();
    cv::waitKey(0);
}

void TestExtractHLine() {
    Image input_image("../images/hv_line.bmp");
    if (input_image.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.ExtractHorizontalAndVeticalLine(input_image, false);
    input_image.ShowSrcImage();
    input_image.ShowDstImage();
    cv::waitKey(0);
}

void TestExtractChars() {
    Image input_image("../images/chars.png");
    if (input_image.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.ExtractChars(input_image);
    input_image.ShowSrcImage();
    input_image.ShowDstImage();
    cv::waitKey(0);
}

void TestPyrUp() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.PyramidUp(lena);
    lena.ShowSrcImage();
    lena.ShowDstImage();
}

void TestPyrDown() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.PyramidDown(lena);
    lena.ShowSrcImage();
    lena.ShowDstImage();
}

void TestDog() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.Dog(lena);
    lena.ShowSrcImage();
    lena.ShowDstImage();
}

int threshold_type = 0;
void ThresholdTrackbarCallback(int pos, void* userdata) {
    if (userdata == nullptr) {
        return;
    }
    Image& image = std::ref(*(static_cast<Image*>(userdata)));
    ImageProcessor processor;
    const int threshold_max = 255;
    processor.ThresholdOperation(image, pos, threshold_max, threshold_type);
    image.ShowDstImage();
}

void TestThreshold() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    lena.ShowSrcImage();
    cv::cvtColor(lena.GetSrcImage(), lena.GetDstImage(), cv::COLOR_BGR2GRAY);
    cv::swap(lena.GetSrcImage(), lena.GetDstImage());

    int threshold_value = 127;
    const int threshold_max = 255;
    const int threshold_type_max = 4;
    cv::createTrackbar("threshold value", lena.GetOutputWindowName(),
        &threshold_value, threshold_max, ThresholdTrackbarCallback, &lena);
    cv::createTrackbar("threshold type", lena.GetOutputWindowName(),
        &threshold_type, threshold_type_max, ThresholdTrackbarCallback, &lena);
    ThresholdTrackbarCallback(threshold_value, &lena);

    cv::waitKey(0);
}

void TestRobertKernel() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    //processor.RobertKernelX(lena);
    processor.RobertKernelY(lena);
    lena.ShowDstImage();
    cv::waitKey(0);
}

void TestSobelKernel() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.SobelKernelX(lena);
    lena.ShowDstImage();
    cv::waitKey(0);
}

void TestLaplaceKernel() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.LaplaceKernel(lena);
    lena.ShowDstImage();
    cv::waitKey(0);
}

void TestBorder() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    // cv::BORDER_DEFAULT£¬ cv::BORDER_WRAP, cv::BORDER_REPLICATE£¬cv::BORDER_CONSTANT
    int border_type = cv::BORDER_WRAP;
    int bottom, top;
    bottom = top = lena.GetSrcImage().rows * 0.05;
    int left, right;
    left = right = lena.GetSrcImage().cols * 0.05;

    cv::RNG rng;
    //for (auto i = 0; i < 6; ++i) {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::copyMakeBorder(lena.GetSrcImage(), lena.GetDstImage(), top, bottom, left, right, border_type, color);
        lena.ShowDstImage();
    //}
    
    
    cv::waitKey(0);
}

void TestSobelGradient() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    lena.ShowSrcImage();
    ImageProcessor processor;
    processor.SobelGradient(lena);
    lena.ShowDstImage();
    cv::waitKey(0);
}

void TestLaplacian() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    lena.ShowSrcImage();
    ImageProcessor processor;
    processor.Laplacian(lena);
    lena.ShowDstImage();
    cv::waitKey(0);
}

void CannyTrackbarCallback(int pos, void *user_data) {
    if (user_data == nullptr) {
        return;
    }
    Image& image = std::ref(*(static_cast<Image*>(user_data)));
    ImageProcessor processor;
    processor.CannyEdgeDetection(image, pos, pos *2);

    image.ShowDstImage();
}

void TestCanny() {
    Image lena("../images/lena.jpg");
    if (lena.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }
    lena.ShowSrcImage();

    int init_value = 50;
    cv::createTrackbar("threshold value", lena.GetOutputWindowName(),
        &init_value, 2 * init_value, CannyTrackbarCallback, &lena);
    CannyTrackbarCallback(init_value, &lena);

    cv::waitKey(0);
}

void TestHoughLine() {
    Image input_image("../images/lena.jpg");
    if (input_image.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }
    input_image.ShowSrcImage();

    ImageProcessor processor;
    processor.HoughLineDetection(input_image);

    input_image.ShowDstImage();
    cv::waitKey(0);
}

void TestHoughCircle() {
    Image input_image("../images/circle.bmp");
    if (input_image.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }
    input_image.ShowSrcImage();

    ImageProcessor processor;
    processor.HoughCircleDetection(input_image);

    input_image.ShowDstImage();
    cv::waitKey(0);
}

void TestHistogram() {
    Image input_image("../images/lena.jpg");
    if (input_image.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }
    input_image.ShowSrcImage();

    ImageProcessor processor;
    processor.CalculateHistogram(input_image);

    input_image.ShowDstImage();
    cv::waitKey(0);
}

void TestCompareHist() {
    Image first_image("../images/lena.jpg", "first input image", "first output image");
    if (first_image.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    Image second_image("../images/aloeL.jpg", "second input image", "second output image");
    if (second_image.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    int compare_method = cv::HISTCMP_CORREL;
    double ret = processor.CompareHistogram(first_image, second_image, compare_method);

    first_image.GetDstImage() = first_image.GetSrcImage();
    cv::putText(first_image.GetDstImage(), std::to_string(ret), cv::Point(100, 100),
        cv::FONT_HERSHEY_PLAIN, 5, cv::Scalar(0, 255, 0), 2);

    first_image.ShowDstImage();

    second_image.GetDstImage() = second_image.GetSrcImage();
    second_image.ShowDstImage();
    cv::waitKey(0);
}

void TestBackProjHist() {
    Image input_image("../images/lena.jpg", "first input image", "first output image");
    if (input_image.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.BackProjectHist(input_image);
    input_image.ShowSrcImage();
    input_image.ShowDstImage();

    cv::waitKey(0);
}

void TestTemplateMatch() {
    Image input_image("../images/lena.jpg", "first input image", "first output image");
    if (input_image.Empty()) {
        LOG_ERROR << "Can't read image from given path." << std::endl;
        return;
    }

    cv::Mat template_img = cv::imread("../images/lena_template.png");
    if (template_img.empty()) {
        LOG_ERROR << "Can't read template image." << std::endl;
        return;
    }

    int match_method = cv::TM_SQDIFF;//cv::TM_CCOEFF;
    ImageProcessor processor;
    processor.TemplateMatch(input_image, template_img, match_method);

    input_image.ShowSrcImage();
    input_image.ShowDstImage();
    
    cv::waitKey(0);
}

void FindContoursCallback(int pos, void* user_data) {
    if (user_data == nullptr) {
        return;
    }
    Image& image = std::ref(*(static_cast<Image*>(user_data)));
    ImageProcessor processor;
    processor.DiscoverContours(image, pos);

    image.ShowDstImage();
}

void TestFindContours() {
    Image input_image("../images/lena.jpg", "first input image", "first output image");
    if (input_image.Empty()) {
        LOG_ERROR << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.DiscoverContours(input_image, 50);

    input_image.ShowSrcImage();

    int pos = 90;
    cv::createTrackbar("threshold value", input_image.GetOutputWindowName(),
        &pos, 255, FindContoursCallback, &input_image);
    FindContoursCallback(pos, &input_image);

    cv::waitKey(0);
}

void TestConvexHull() {
    Image input_image("../images/lena.jpg", "first input image", "first output image");
    if (input_image.Empty()) {
        LOG_ERROR << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.FindConvexHull(input_image);

    input_image.ShowSrcImage();
    input_image.ShowDstImage();

    cv::waitKey(0);
}

void TestFindTarget() {
    Image input_image("../images/hot.png", "first input image", "first output image");
    if (input_image.Empty()) {
        LOG_ERROR << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    processor.FindTarget(input_image);

    input_image.ShowSrcImage();
    input_image.ShowDstImage();

    cv::waitKey(0);
}

void TestCalculateMoments() {
    Image input_image("../images/hot.png", "first input image", "first output image");
    if (input_image.Empty()) {
        LOG_ERROR << "Can't read image from given path." << std::endl;
        return;
    }

    ImageProcessor processor;
    int threshold_value = 56;
    processor.CalculateMoments(input_image, threshold_value);

    input_image.ShowSrcImage();
    input_image.ShowDstImage();

    cv::waitKey(0);
}

}