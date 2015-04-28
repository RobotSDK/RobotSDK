#-------------------------------------------------
#
# Project created by QtCreator 2015-04-27T19:38:35
#
#-------------------------------------------------

QT       -= gui

TARGET = VirtualScan
TEMPLATE = lib

SOURCES += \
    VirtualScanGenerator.cpp \
    VirtualScanViewer.cpp \
    VirtualScanPublisher.cpp \
    VirtualScanCluster.cpp

HEADERS += \
    VirtualScanGenerator.h \
    VirtualScanViewer.h \
    VirtualScanPublisher.h \
    VirtualScanCluster.h

MODULES += Velodyne
include($$(HOME)/SDK/RobotSDK_4.0/Kernel/RobotSDK.pri)

unix{
    INCLUDEPATH += /usr/include/pcl-1.7

    LIBS += -L/usr/lib -lpcl_io
    LIBS += -L/usr/lib -lpcl_common
    LIBS += -L/usr/lib -lpcl_filters
    LIBS += -L/usr/lib -lpcl_search
    LIBS += -L/usr/lib -lpcl_kdtree
    LIBS += -L/usr/lib -lpcl_features
    LIBS += -L/usr/lib -lpcl_segmentation

    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_highgui
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_features2d
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_objdetect
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_contrib
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_imgproc

    INCLUDEPATH += $$(HOME)/SDK/FastVirtualScan/include

    CONFIG(debug, debug|release){
        LIBS += -L$$(HOME)/SDK/FastVirtualScan/lib/ -lFastVirtualScan_Debug
    }else{
        LIBS += -L$$(HOME)/SDK/FastVirtualScan/lib/ -lFastVirtualScan_Release
    }
}
