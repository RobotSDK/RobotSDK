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
    VirtualScanCluster.cpp \
    fastvirtualscan.cpp

HEADERS += \
    VirtualScanGenerator.h \
    VirtualScanViewer.h \
    VirtualScanPublisher.h \
    VirtualScanCluster.h \
    fastvirtualscan.h

MODULES += Velodyne
include($$(HOME)/SDK/RobotSDK_4.0/Kernel/RobotSDK.pri)

unix{
    INCLUDEPATH += /usr/local/include/pcl-1.8
    LIBS += -L/usr/local/lib -lpcl_io
    LIBS += -L/usr/local/lib -lpcl_common
    LIBS += -L/usr/local/lib -lpcl_filters
    LIBS += -L/usr/local/lib -lpcl_search
    LIBS += -L/usr/local/lib -lpcl_kdtree
    LIBS += -L/usr/local/lib -lpcl_features
    LIBS += -L/usr/local/lib -lpcl_segmentation

    INCLUDEPATH += /usr/local/include
    LIBS += -L/usr/local/lib -lopencv_core
    LIBS += -L/usr/local/lib -lopencv_highgui
    LIBS += -L/usr/local/lib -lopencv_features2d
    LIBS += -L/usr/local/lib -lopencv_objdetect
    LIBS += -L/usr/local/lib -lopencv_contrib
    LIBS += -L/usr/local/lib -lopencv_imgproc
}
