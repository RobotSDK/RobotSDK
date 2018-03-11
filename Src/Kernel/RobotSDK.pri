QT *= core widgets xml opengl
CONFIG *= c++11 qt
DEFINES *= RobotSDK_ModDev
#DEFINES *= RobotSDK_Debug

INCLUDEPATH += .

ROBOTSDKVER=4.0

unix{
    isEmpty(MODULES){
    }
    else{
        for(module, MODULES){
            INCLUDEPATH += $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Module/$$module/include
        }
    }

    INCLUDEPATH += $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/include
    CONFIG(debug, debug|release){
        LIBS += -L$$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/lib -lKernel_Debug
    }
    else{
        LIBS += -L$$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/lib -lKernel_Release
    }

    INCLUDEPATH += /usr/include/eigen3

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
        LIBS *= /usr/lib/x86_64-linux-gnu/libboost_system.so
        INCLUDEPATH += /opt/ros/$$(ROS_DISTRO)/include
    }
    LIBS *= -L/usr/lib/x86_64-linux-gnu -lglut -lGLU

    MOC_DIR = $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/MOC
    UI_DIR = $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/UI

    CONFIG(debug, debug|release){
        OBJECTS_DIR = $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/OBJ/Debug
        DESTDIR = $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/lib/Debug
        target.path = $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Module/$$TARGET/lib/Debug
    }
    else{
        OBJECTS_DIR = $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/OBJ/Release
        DESTDIR = $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/lib/Release
        target.path = $$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Module/$$TARGET/lib/Release
    }
    INSTALLS += target

    headers.path=$$(HOME)/SDK/RobotSDK_$${ROBOTSDKVER}/Module/$$TARGET/include
    headers.files=$$HEADERS    
    INSTALLS += headers
}

win32{
    isEmpty(MODULES){
    }
    else{
        for(module, MODULES){
            INCLUDEPATH += C:/SDK/RobotSDK_$${ROBOTSDKVER}/Module/$$module/include
        }
    }

    INCLUDEPATH += C:/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/include
    CONFIG(debug, debug|release){
        LIBS += -LC:/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/lib -lKernel_Debug
    }
    else{
        LIBS += -LC:/SDK/RobotSDK_$${ROBOTSDKVER}/Kernel/lib -lKernel_Release
    }
    EIGEN=$$(EIGEN_PATH)
    isEmpty(EIGEN){
    }
    else{
        INCLUDEPATH += $$(EIGEN_PATH)
    }

    MOC_DIR = C:/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/MOC
    UI_DIR = C:/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/UI

    CONFIG(debug, debug|release){
        OBJECTS_DIR = C:/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/OBJ/Debug
        DESTDIR = C:/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/lib/Debug
        target.path = C:/SDK/RobotSDK_$${ROBOTSDKVER}/Module/$$TARGET/lib/Debug
    }
    else{
        OBJECTS_DIR = C:/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/OBJ/Release
        DESTDIR = C:/SDK/RobotSDK_$${ROBOTSDKVER}/Build/Module/$$TARGET/lib/Release
        target.path = C:/SDK/RobotSDK_$${ROBOTSDKVER}/Module/$$TARGET/lib/Release
    }
    INSTALLS += target

    headers.path=C:/SDK/RobotSDK_$${ROBOTSDKVER}/Module/$$TARGET/include
    headers.files=$$HEADERS
    INSTALLS += headers
}
