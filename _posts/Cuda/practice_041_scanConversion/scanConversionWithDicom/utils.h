#pragma once
#include <windows.h>
#include <vector>
#include <string>
#include <iostream>


std::string GetExecutableParentDirectory();
std::string GetParentDirectory(const std::string& directoryPath);
std::string GetResourceDirectory(std::string& resourceFolderName);
std::pair<std::string, std::vector<std::string>> FindFolderAndFileListByNameRecursively(const std::string& directory, const std::string& folderName);
std::vector<std::string> ListFilesInFolder(const std::string& folderPath);

