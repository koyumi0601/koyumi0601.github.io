#include <iostream>
#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmdata/dctk.h>

int main()
{
    DcmFileFormat dcmFile;
    OFCondition status = dcmFile.loadFile("path/to/your/dicom/file.dcm");

    if (status.good())
    {
        std::cout << "DICOM file loaded successfully!" << std::endl;
        // Do something with the DICOM file
    }
    else
    {
        std::cerr << "Error loading DICOM file: " << status.text() << std::endl;
    }

    return 0;
}