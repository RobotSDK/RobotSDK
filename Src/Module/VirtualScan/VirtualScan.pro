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
include($$(ROBOTSDKMODULE))

unix{
    INCLUDEPATH += /usr/local/include/pcl-1.8
    LIBS += -L/usr/local/lib -lpcl_io
    LIBS += -L/usr/local/lib -lpcl_common
    LIBS += -L/usr/local/lib -lpcl_filters
    LIBS += -L/usr/local/lib -lpcl_search
    LIBS += -L/usr/local/lib -lpcl_kdtree
    LIBS += -L/usr/local/lib -lpcl_features
    LIBS += -L/usr/local/lib -lpcl_segmentation

    INCLUDEPATH += /usr/include
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_core
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_highgui
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_features2d
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_objdetect
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_contrib
    LIBS += -L/usr/lib/x86_64-linux-gnu -lopencv_imgproc
}
