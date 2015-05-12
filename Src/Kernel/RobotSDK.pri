QT *= widgets xml opengl
CONFIG *= c++11 qt
DEFINES *= RobotSDK_ModDev

INCLUDEPATH += .

unix{
    isEmpty(MODULES){
    }
    else{
        for(module, MODULES){
            INCLUDEPATH += $$(HOME)/SDK/RobotSDK_4.0/Module/$$module/include
        }
    }

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
            LIBS *= -L/opt/ros/indigo/lib -lcv_bridge
            LIBS *= -L/usr/lib/x86_64-linux-gnu -lboost_system
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
        LIBS *= -L/opt/ros/indigo/lib -lcv_bridge
        LIBS *= -L/usr/lib/x86_64-linux-gnu -lboost_system
        INCLUDEPATH += /opt/ros/$$(ROS_DISTRO)/include
    }

    MOC_DIR = $$(HOME)/SDK/RobotSDK_4.0/Build/Module/$$TARGET/MOC
    UI_DIR = $$(HOME)/SDK/RobotSDK_4.0/Build/Module/$$TARGET/UI

    CONFIG(debug, debug|release){
        OBJECTS_DIR = $$(HOME)/SDK/RobotSDK_4.0/Build/Module/$$TARGET/OBJ/Debug
        DESTDIR = $$(HOME)/SDK/RobotSDK_4.0/Build/Module/$$TARGET/lib/Debug
        target.path = $$(HOME)/SDK/RobotSDK_4.0/Module/$$TARGET/lib/Debug
    }
    else{
        OBJECTS_DIR = $$(HOME)/SDK/RobotSDK_4.0/Build/Module/$$TARGET/OBJ/Release
        DESTDIR = $$(HOME)/SDK/RobotSDK_4.0/Build/Module/$$TARGET/lib/Release
        target.path = $$(HOME)/SDK/RobotSDK_4.0/Module/$$TARGET/lib/Release
    }
    INSTALLS += target

    headers.path=$$(HOME)/SDK/RobotSDK_4.0/Module/$$TARGET/include
    headers.files=$$HEADERS    
    INSTALLS += headers
}

win32{
    isEmpty(MODULES){
    }
    else{
        for(module, MODULES){
            INCLUDEPATH += C:/SDK/RobotSDK_4.0/Module/$$module/include
        }
    }

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

    MOC_DIR = C:/SDK/RobotSDK_4.0/Build/Module/$$TARGET/MOC
    UI_DIR = C:/SDK/RobotSDK_4.0/Build/Module/$$TARGET/UI

    CONFIG(debug, debug|release){
        OBJECTS_DIR = C:/SDK/RobotSDK_4.0/Build/Module/$$TARGET/OBJ/Debug
        DESTDIR = C:/SDK/RobotSDK_4.0/Build/Module/$$TARGET/lib/Debug
        target.path = C:/SDK/RobotSDK_4.0/Module/$$TARGET/lib/Debug
    }
    else{
        OBJECTS_DIR = C:/SDK/RobotSDK_4.0/Build/Module/$$TARGET/OBJ/Release
        DESTDIR = C:/SDK/RobotSDK_4.0/Build/Module/$$TARGET/lib/Release
        target.path = C:/SDK/RobotSDK_4.0/Module/$$TARGET/lib/Release
    }
    INSTALLS += target

    headers.path=C:/SDK/RobotSDK_4.0/Module/$$TARGET/include
    headers.files=$$HEADERS
    INSTALLS += headers
}
