#pragma once

#include <string>

#include "logger.h"

class Utility
{
public:
    static bool PathExists(const std::string& path);
    static std::string BackSlash2ForwardSlash(const std::string& path);
    static bool InitializeGlog(const std::string& path = "/log");

private:
    static bool CheckFolderExist(const std::string& path);
};

