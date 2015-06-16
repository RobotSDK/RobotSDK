#-------------------------------------------------
#
# Project created by QtCreator 2015-06-12T19:42:24
#
#-------------------------------------------------

QT       -= gui

TARGET = GPUTracker
TEMPLATE = lib

DEFINES += GPUTRACKER_LIBRARY

SOURCES += \
    ObstacleMapGenerator.cu \
    ObstacleMapGenerator.cpp \
    ObstacleMapViewer.cpp \
    ObstacleMapGlobalizer.cpp \
    ObjectDetection.cpp \
    ObjectParticleResample.cpp \
    ObjectDetectionWidget.cpp

HEADERS += \
    ObstacleMapGenerator.cuh \
    ObstacleMapGenerator.h \
    ObstacleMapViewer.h \
    ObstacleMapGlobalizer.h \
    ObjectDetection.h \
    ObjectParticleResample.h \
    ObjectDetectionWidget.h

MODULES += VirtualScan Velodyne Localization
include($$(ROBOTSDKMODULE))
include($$(ROBOTSDKCUDA))

unix {
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
