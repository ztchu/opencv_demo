#include "image_test.h"

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

}