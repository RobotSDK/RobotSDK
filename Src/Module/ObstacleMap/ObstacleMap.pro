#-------------------------------------------------
#
# Project created by QtCreator 2015-06-02T17:15:39
#
#-------------------------------------------------

QT       -= gui

TARGET = ObstacleMap
TEMPLATE = lib

SOURCES += \
    ObstacleMapGenerator.cpp \
    ObstacleMapViewer.cpp \
    ObstacleMapGenerator.cu \
    TrackingParticleGenerator.cpp \
    ObstacleMapMeasure.cpp

HEADERS += \
    ObstacleMapGenerator.h \
    ObstacleMapViewer.h \
    ObstacleMapGenerator.cuh \
    TrackingParticleGenerator.h \
    ObstacleMapMeasure.h

MODULES += VirtualScan Velodyne
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

