#-------------------------------------------------
#
# Project created by QtCreator 2015-07-06T11:29:26
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = TestCornerDetector
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h \
    egotransform.h \
    particlefilterbase.h \
    particlefilterdef.h \
    randomgenerator.h

FORMS    += mainwindow.ui

DISTFILES += \
    egotransform.cu

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


    INCLUDEPATH += $$(HOME)/SDK/RobotSDK_4.0/Kernel/include
    CONFIG(debug, debug|release){
        LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
    }
    else{
        LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Release
    }
}

ROS = $$(ROS_DISTRO)
isEmpty(ROS){
    error(Please install ROS first or run via terminal if you have ROS installed)
}
else{
    LIBS *= -L/opt/ros/$$ROS/lib -lroscpp
    LIBS *= -L/opt/ros/$$ROS/lib -lrosconsole
    LIBS *= -L/opt/ros/$$ROS/lib -lroscpp_serialization
    LIBS *= -L/opt/ros/$$ROS/lib -lrostime
    LIBS *= -L/opt/ros/$$ROS/lib -lxmlrpcpp
    LIBS *= -L/opt/ros/$$ROS/lib -lcpp_common
    LIBS *= -L/opt/ros/$$ROS/lib -lrosconsole_log4cxx
    LIBS *= -L/opt/ros/$$ROS/lib -lrosconsole_backend_interface
    LIBS *= -L/opt/ros/$$ROS/lib -ltf
    LIBS *= -L/opt/ros/$$ROS/lib -ltf2
    LIBS *= -L/opt/ros/$$ROS/lib -ltf2_ros
    LIBS *= -L/opt/ros/$$ROS/lib -lpcl_ros_tf
    LIBS *= -L/opt/ros/$$ROS/lib -ltf_conversions
    LIBS *= -L/opt/ros/$$ROS/lib -lactionlib
    LIBS *= -L/opt/ros/$$ROS/lib -lcv_bridge
    LIBS *= -L/opt/ros/$$ROS/lib -lrosbag
    LIBS *= -L/opt/ros/$$ROS/lib -lrosbag_storage
    LIBS *= -L/usr/lib/x86_64-linux-gnu -lboost_system
    INCLUDEPATH += /opt/ros/$$(ROS_DISTRO)/include
}
