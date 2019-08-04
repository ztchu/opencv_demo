#include <iostream>


#include "Image.h"
#include "ImageProcessor.h"
#include "logger.h"
#include "Utility.h"

#define TEST_MASK_OPERATION 1
#define TEST_FILTER2D 0

int main(int argc, char** argv) { 
#ifdef GLOG
    if (!Utility::InitializeGlog()) {
        std::cerr << "Can't init glog";
    }
#endif
    Image aloe("../images/aloeL.jpg");
    if (aloe.Empty()) {
        std::cerr << "Can't read image from given path." << std::endl;
        return -1;
    }
    aloe.ShowSrcImage();

    ImageProcessor processor;
#if TEST_MASK_OPERATION
    processor.MaskOperation(aloe);
#endif
#if TEST_FILTER2D
    cv::Mat kernel = (cv::Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    processor.SetKernel(kernel);
    processor.Filter2DMask(aloe);
#endif
    aloe.ShowDstImage();
    
    cv::waitKey(0);
#ifdef GLOG
    google::ShutdownGoogleLogging();
#endif
}