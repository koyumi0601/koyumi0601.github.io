#include "utils.h"


std::string GetExecutableParentDirectory() {
  char path[MAX_PATH] = { 0 };
  GetModuleFileNameA(NULL, path, MAX_PATH);
  std::string fullPath(path);
  size_t pos = fullPath.find_last_of("\\/");
  std::string parentDirectory = (pos != std::string::npos) ? fullPath.substr(0, pos) : "";
  return parentDirectory;
}

std::string GetParentDirectory(const std::string& directoryPath) {
  size_t pos = directoryPath.find_last_of("\\/");
  std::string parentDirectory = (pos != std::string::npos) ? directoryPath.substr(0, pos) : "";
  return parentDirectory;
}

std::string GetResourceDirectory(std::string& resourceFolderName) {
  std::string exeParentDirectory = GetExecutableParentDirectory();
  std::string firstParentDirectory = GetParentDirectory(exeParentDirectory);
  std::string secondParentDirectory = GetParentDirectory(firstParentDirectory);
  std::string resourcePath = secondParentDirectory + "\\" + resourceFolderName;
  return resourcePath;
}

std::pair<std::string, std::vector<std::string>> FindFolderAndFileListByNameRecursively(const std::string& directory, const std::string& folderName) {

  std::string searchPath = directory + "\\*";
  WIN32_FIND_DATA findData;
  HANDLE hFind = FindFirstFile(searchPath.c_str(), &findData);

  if (hFind == INVALID_HANDLE_VALUE) {
    std::cerr << "Failed to access directory: " << directory << std::endl;
    return { "", {} };
  }
  do {
    if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
      // Skip "." and ".." directories
      if (strcmp(findData.cFileName, ".") != 0 && strcmp(findData.cFileName, "..") != 0) {
        std::string currentDirName = findData.cFileName;
        std::string currentPath = directory + "\\" + currentDirName;
        if (currentDirName == folderName) {
          FindClose(hFind);
          std::vector<std::string> fileList = ListFilesInFolder(currentPath);
          return { currentPath, fileList };
        }
        else {
          // Recursive call to search in subdirectories
          std::pair<std::string, std::vector<std::string>> found = FindFolderAndFileListByNameRecursively(currentPath, folderName);
          if (!found.first.empty()) {
            FindClose(hFind);
            return found;
          }
        }
      }
    }
  } while (FindNextFile(hFind, &findData) != 0);
  FindClose(hFind);
  return { "", {} };
}

std::vector<std::string> ListFilesInFolder(const std::string& folderPath) {
  std::vector<std::string> fileList;
  WIN32_FIND_DATA findData;
  HANDLE hFind = FindFirstFile((folderPath + "\\*").c_str(), &findData);
  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        fileList.push_back(findData.cFileName);
      }
    } while (FindNextFile(hFind, &findData) != 0);
    FindClose(hFind);
  }
  return fileList;
}

