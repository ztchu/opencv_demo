#include "Utility.h"

#include <direct.h>
#include <fstream>
#include <iostream>
#include <Windows.h>


#define MAXPATH 1000

bool Utility::PathExists(const std::string& path) {
    std::ifstream fin(path.c_str());
    if (fin.is_open()) {
        std::cout << "File exists" << std::endl;
        return true;
    }
    else {
        if (Utility::CheckFolderExist(path)) {
            std::cout << "Directory exists" << std::endl;
            return true;
        }
        std::cout << "Path not exist" << std::endl;
        return false;
    }
}

bool Utility::CheckFolderExist(const std::string& path)
{
    WIN32_FIND_DATA  find_file_data;
    bool ret_value = false;
    HANDLE find_handle = FindFirstFile(path.c_str(), &find_file_data);
    if ((find_handle != INVALID_HANDLE_VALUE) && (find_file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
    {
        ret_value = true;
    }
    FindClose(find_handle);
    return ret_value;
}

std::string Utility::BackSlash2ForwardSlash(const std::string& path) {
    std::string result(path.size(), '\0');
    for (size_t i = 0, j = 0; i < path.size(); ++j, ++i) {
        if (path[i] == '\\') {
            if (i + 1 < path.size() && path[i + 1] == '\\') {
                ++i;
            }
            result[j] = '/';
        }
        else {
            result[j] = path[i];
        }
    }
    return result;
}

#ifdef GLOG
bool Utility::InitializeGlog(const std::string& path) {
    char buffer[MAXPATH];
    if (_getcwd(buffer, MAXPATH) == nullptr) {
        std::cerr << "Get current path failed.";
        return false;
    }

    std::string log_dir(buffer);
    log_dir.append(path);

    if (!Utility::PathExists(log_dir)) {
        std::string temp_dir("\"" + log_dir + "\"");
        temp_dir = "mkdir " + temp_dir;
        std::string form_cmd_line = Utility::BackSlash2ForwardSlash(temp_dir);
        system(form_cmd_line.c_str());
    }

    FLAGS_log_dir = log_dir.c_str();
    google::InitGoogleLogging(log_dir.c_str());
    google::SetLogDestination(google::GLOG_INFO, (log_dir + "/info_").c_str());
    google::SetLogDestination(google::GLOG_WARNING, (log_dir + "/warn_").c_str());
    google::SetLogDestination(google::GLOG_ERROR, (log_dir + "/error_").c_str());
    google::SetLogFilenameExtension(".log");
    FLAGS_stderrthreshold = google::GLOG_INFO;
    FLAGS_colorlogtostderr = true;  // Set log color

    return true;
}
#endif