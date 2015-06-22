#ifndef VEHICLETRACKER
#define VEHICLETRACKER

//=================================================
//Please add headers here:
#include<VehicleDetector.h>
#include<ObstacleMapGlobalizer.h>
#include<VehicleParticleFilter.h>
#include<sync.h>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS VehicleTracker

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 2

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{

};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(int, particlenum, 5000)
    ADD_VAR(float, state_x_min, -3)
    ADD_VAR(float, state_x_max, 3)
    ADD_VAR(float, state_y_min, -3)
    ADD_VAR(float, state_y_max, 3)
    ADD_VAR(float, state_theta_min,-0.3)
    ADD_VAR(float, state_theta_max,0.3)
    ADD_VAR(float, state_theta_sigma,0.3)
    ADD_VAR(float, state_v_min, -10)
    ADD_VAR(float, state_v_max, 10)
    ADD_VAR(float, state_v_sigma, 5)
    ADD_VAR(float, state_width_min, -2)
    ADD_VAR(float, state_width_max, 2)
    ADD_VAR(float, state_length_min, -2)
    ADD_VAR(float, state_length_max, 2)
    ADD_VAR(float, threshold, 1.0)
public:
    ADD_VAR(int, obmap_edgepointnum, 10)
    ADD_VAR(float, obmap_margin, 0.1)
    ADD_VAR(QString, obmap_wtable, "2.0,1.0,0.4")
    MEASUREDATA_TYPE(Vehicle) measuredata;
public:
    ADD_SYNC(mapsync,1)
public:
    QTime curtimestamp;
    cv::Mat curtransform;
    float curtheta;
    cv::Mat localheadvec;
    std::vector<int> objectid;
    std::vector<STATE_TYPE(Vehicle)> objectstate;
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    cv::Mat transform;
    std::vector<int> objectid;
    std::vector<STATE_TYPE(Vehicle)> objectstate;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
