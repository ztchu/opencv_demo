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

}