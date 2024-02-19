
#include <iostream>
#include <string>
#include <../src/database/database_manager.h>
#include <../src/imaging_system/image_filter.h>
#include <../src/imaging_system/image_loader.h>
#include <../src/imaging_system/image_transform.h>
#include <../src/imaging_system/ocr.h>
#include <../src/networking/network_manager.h>
#include <../src/userinterface/mainwindow.h>
#include <../src/logger/logger.h>




int main()
{

    // 정적할당: 컴파일 시간에 메모리가 할당되고 해제되는 것
    // 지역변수 혹은 전역변수
    // imgLoader imgLoaderObject;
    // imgLoaderObject.printImageLoader(); // 동적으로 할당된 객체에 접근할 때에는 클래스의 인스턴스를 직접 참조하기 때문에 . 연산자를 통해 멤버에 접근한다.
    
    // 동적할당: 프로그램 실행 중 메모리를 할당하고 해제하는 것
    // 포인터를 이용하여 메모리에 접근 및 조작

    // std::cout << "This is main\n";

    databaseManager* dynamicDbMgrPtr = new databaseManager();
    dynamicDbMgrPtr->printDatabaseManager();
    dynamicDbMgrPtr->createDatabase();
    dynamicDbMgrPtr->insertDatabase();
    dynamicDbMgrPtr->viewDatabase();
    delete dynamicDbMgrPtr;

    // imageFilter* dynamicImgFiltPtr = new imageFilter();
    // dynamicImgFiltPtr->printImageFilter();
    // delete dynamicImgFiltPtr;
    
    // imgLoader* dynamicImgLoaderPtr = new imgLoader();
    // dynamicImgLoaderPtr->printImageLoader();
    // dynamicImgLoaderPtr->loadAndShow();
    // delete dynamicImgLoaderPtr;

    // imageTransform* dynamicImageTransformPtr = new imageTransform();
    // dynamicImageTransformPtr->printImageTransform();
    // delete dynamicImageTransformPtr;

    // ocr* dynamicOcrPtr = new ocr();
    // dynamicOcrPtr->printOcr();
    // delete dynamicOcrPtr;

    // networkManager* dynamicNetworkManagerPtr = new networkManager();
    // dynamicNetworkManagerPtr->printNetworkManager();
    // // dynamicNetworkManagerPtr->openWeb(); // test 
    // delete dynamicNetworkManagerPtr;

    // mainwindow* dynamicMainwindowPtr = new mainwindow();
    // dynamicMainwindowPtr->printMainwindow();
    // delete dynamicMainwindowPtr;

    // // logger: singletone designed
    // logger& loggerInstance = logger::getInstance();
    // loggerInstance.printLogger();
    
    // 프로그램이 종료되면 정적으로 할당된 클래스들의 인스턴스의 destructor가 자동으로 실행 됨.
    return 0;
        
}