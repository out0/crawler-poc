#ifndef _FILE_UTILS_H
#define _FILE_UTILS_H

#include <string>
#include <fstream>
#include <sys/stat.h>

class FileData {
public:
    char *data;
    size_t length;

    FileData(int length) {
        this->length = length;
        this->data = new char[length];
    }
};

class FileUtils
{
public:
    //     unsigned char *result = new unsigned char[length];
    //     for (int i =0; i < length; i++) {
    //         result[i] = abs(buffer[i]);
    //     }

    static FileData *readFile(std::string filename)
    {
        std::ifstream file(filename, std::ifstream::binary);
        file.seekg(0, file.end);
        size_t length = static_cast<size_t>(file.tellg());
        file.seekg(0, file.beg);

        FileData *buffer = new FileData(length);
        file.read(buffer->data, length);
        file.close();

        return buffer;
    }

    static inline bool fileExists(const std::string &name)
    {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }
};

#endif