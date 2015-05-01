#ifndef IMAGEPOINTCLOUDFUSIONVIEWER
#define IMAGEPOINTCLOUDFUSIONVIEWER

//=================================================
//Please add headers here:
#include"CameraVelodyneFusion.h"
#include<QLabel>
#include<QScrollArea>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS ImagePointCloudFusionViewer

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 0

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public RobotSDK::NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(double, minrange, 0)
    ADD_PARAM(double, maxrange, 100)
    ADD_ENUM_PARAM_WITH_OPTIONS(int, colormapid, cv::COLORMAP_JET, QList<int>()
                                <<cv::COLORMAP_AUTUMN
                                <<cv::COLORMAP_BONE
                                <<cv::COLORMAP_JET
                                <<cv::COLORMAP_WINTER
                                <<cv::COLORMAP_RAINBOW
                                <<cv::COLORMAP_OCEAN
                                <<cv::COLORMAP_SUMMER
                                <<cv::COLORMAP_SPRING
                                <<cv::COLORMAP_COOL
                                <<cv::COLORMAP_HSV
                                <<cv::COLORMAP_PINK
                                <<cv::COLORMAP_HOT)
};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public RobotSDK::NODE_VARS_BASE_TYPE
{
public:
    QVector<QRgb> colortable;
    cv::Mat colormap;
public:
    ADD_QLAYOUT(QHBoxLayout, layout)
    ADD_QWIDGET(QTabWidget, tabwidget)
    ADD_QWIDGET(QScrollArea, scrollarea)
    ADD_QWIDGET(QLabel, viewer, "Image & Point Cloud Fusion Viewer")
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public RobotSDK::NODE_DATA_BASE_TYPE
{

};

//=================================================
//You can declare functions here


//=================================================
}

#endif
