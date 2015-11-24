#ifndef NDTLOCALIZER
#define NDTLOCALIZER

//=================================================
//Please add headers here:
#include<opencv2/opencv.hpp>
#include<rosinterface.h>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS NDTLocalizer

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 0

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{

};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(QString, calibfilename, "#(NDT3DCalibFilename)")
public:
    cv::Mat extrinsicmat;
public:
    ADD_VAR(QString, destinationframe, "world")
    ADD_VAR(QString, originalframe, "ndt_frame")
    ADD_VAR(int, queryinterval, 10)
public:
    ADD_INTERNAL_QOBJECT_TRIGGER(ROSTFSub, ndtsub, 1, destinationframe, originalframe, queryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(ndtsub, receiveTFSignal)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    cv::Mat cvtransform;
public:
    cv::Mat extrinsicmat;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
