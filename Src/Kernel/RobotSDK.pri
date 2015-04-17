QT *= widgets xml opengl
CONFIG *= c++11
DEFINES *= RobotSDK_Module

INCLUDEPATH += .

unix{
    INCLUDEPATH += $$(HOME)/SDK/RobotSDK_4.0/Kernel/include
    CONFIG(debug, debug|release){
        LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
    }
    else{
        LIBS += -L$$(HOME)/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Release
    }

    INCLUDEPATH += /usr/include/eigen3
    LIBS *= -L/usr/lib/i386-linux-gnu -lGLU

    ROS = $$(ROS_DISTRO)
    isEmpty(ROS){
        isEmpty(TMPROS){
        }
        else
        {
            LIBS *= -L/opt/ros/indigo/lib -lroscpp
            LIBS *= -L/opt/ros/indigo/lib -lrosconsole
            LIBS *= -L/opt/ros/indigo/lib -lroscpp_serialization
            LIBS *= -L/opt/ros/indigo/lib -lrostime
            LIBS *= -L/opt/ros/indigo/lib -lxmlrpcpp
            LIBS *= -L/opt/ros/indigo/lib -lcpp_common
            LIBS *= -L/opt/ros/indigo/lib -lrosconsole_log4cxx
            LIBS *= -L/opt/ros/indigo/lib -lrosconsole_backend_interface
            LIBS *= -L/opt/ros/indigo/lib -ltf
            LIBS *= -L/opt/ros/indigo/lib -ltf2
            LIBS *= -L/opt/ros/indigo/lib -ltf2_ros
            LIBS *= -L/opt/ros/indigo/lib -lpcl_ros_tf
            LIBS *= -L/opt/ros/indigo/lib -ltf_conversions
            LIBS *= -L/opt/ros/indigo/lib -lactionlib
            INCLUDEPATH += /opt/ros/$$TMPROS/include
        }
    }
    else{
        LIBS *= -L/opt/ros/indigo/lib -lroscpp
        LIBS *= -L/opt/ros/indigo/lib -lrosconsole
        LIBS *= -L/opt/ros/indigo/lib -lroscpp_serialization
        LIBS *= -L/opt/ros/indigo/lib -lrostime
        LIBS *= -L/opt/ros/indigo/lib -lxmlrpcpp
        LIBS *= -L/opt/ros/indigo/lib -lcpp_common
        LIBS *= -L/opt/ros/indigo/lib -lrosconsole_log4cxx
        LIBS *= -L/opt/ros/indigo/lib -lrosconsole_backend_interface
        LIBS *= -L/opt/ros/indigo/lib -ltf
        LIBS *= -L/opt/ros/indigo/lib -ltf2
        LIBS *= -L/opt/ros/indigo/lib -ltf2_ros
        LIBS *= -L/opt/ros/indigo/lib -lpcl_ros_tf
        LIBS *= -L/opt/ros/indigo/lib -ltf_conversions
        LIBS *= -L/opt/ros/indigo/lib -lactionlib
        INCLUDEPATH += /opt/ros/$$(ROS_DISTRO)/include
    }   
}

win32{
    INCLUDEPATH += C:/SDK/RobotSDK_4.0/Kernel/include
    CONFIG(debug, debug|release){
        LIBS += -LC:/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Debug
    }
    else{
        LIBS += -LC:/SDK/RobotSDK_4.0/Kernel/lib -lKernel_Release
    }
    EIGEN=$$(EIGEN_PATH)
    isEmpty(EIGEN){
    }
    else{
        INCLUDEPATH += $$(EIGEN_PATH)
    }
}
