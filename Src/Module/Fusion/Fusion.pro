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
    ImageDPMFusionViewer.cpp \
    ImageTrackerMarkerFusion.cpp \
    ImageTrackerMarkerFusionViewer.cpp \
    TrackerMarkerReceiver.cpp

HEADERS += \
    CameraVelodyneFusion.h \
    ImagePointCloudFusionViewer.h \
    CameraVirtualScanFusion.h \
    ImageVirtualScanFusionViewer.h \
    CameraDPMFusion.h \
    ImageDPMFusionViewer.h \
    ImageTrackerMarkerFusion.h \
    ImageTrackerMarkerFusionViewer.h \
    TrackerMarkerReceiver.h

MODULES += Camera Velodyne VirtualScan DPM
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

    INCLUDEPATH += $$(HOME)/Git/Autoware/ros/devel/include
}
