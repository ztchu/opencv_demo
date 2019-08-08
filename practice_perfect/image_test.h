#pragma once

#include <iostream>

#include "Image.h"
#include "ImageProcessor.h"
#include "logger.h"
#include "Utility.h"

namespace image_test {
    void TestMaskOperation();
    void TestFilter2D();
    void TestDilate();
    void TestMorphology();
    void TestExtractHLine();
    void TestExtractChars();

    void TestPyrUp();
    void TestPyrDown();
    void TestDog();
}