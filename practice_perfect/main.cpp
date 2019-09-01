
#include "image_test.h"

int main(int argc, char** argv) { 
#ifdef GLOG
    if (!Utility::InitializeGlog()) {
        std::cerr << "Can't init glog";
    }
#endif

    image_test::TestImageSegmentation();

    cv::waitKey(0);

#ifdef GLOG
    google::ShutdownGoogleLogging();
#endif
}
