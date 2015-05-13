#-------------------------------------------------
#
# Project created by QtCreator 2015-04-27T14:44:06
#
#-------------------------------------------------

QT       -= gui

TARGET = Fusion
TEMPLATE = lib

SOURCES += \
    CameraVelodyneFusion.cpp \
    ImagePointCloudFusionViewer.cpp \
    CameraVirtualScanFusion.cpp \
    ImageVirtualScanFusionViewer.cpp \
    CameraDPMFusion.cpp \
    ImageDPMFusionViewer.cpp

HEADERS += \
    CameraVelodyneFusion.h \
    ImagePointCloudFusionViewer.h \
    CameraVirtualScanFusion.h \
    ImageVirtualScanFusionViewer.h \
    CameraDPMFusion.h \
    ImageDPMFusionViewer.h

MODULES += Camera Velodyne VirtualScan DPM
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

    INCLUDEPATH += $$(HOME)/Git/Autoware/ros/devel/include
}
