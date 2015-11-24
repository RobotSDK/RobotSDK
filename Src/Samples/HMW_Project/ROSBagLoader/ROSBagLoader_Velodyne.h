#ifndef ROSBAGLOADER_VELODYNE
#define ROSBAGLOADER_VELODYNE

//=================================================
//Please add headers here:
#include<ros/ros.h>
#include<rosbag/bag.h>
#include<rosbag/view.h>
#include<rosbag/query.h>
#include<sensor_msgs/PointCloud2.h>
#include<tf/tf.h>
#include<tf/tfMessage.h>
#include<geometry_msgs/TransformStamped.h>
#include<rosinterface.h>
#include<QPushButton>
#include<QLabel>
#include<QTimer>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS ROSBagLoader_Velodyne

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 0

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 0

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
    ADD_VAR(QString, bagfile_velodyne, "")
    ADD_VAR(QString, bagfile_tf, "")
    ADD_VAR(QString, bagtopic_velodyne, "/velodyne_points")
    ADD_VAR(QString, bagtopic_tf, "/tf")
    ADD_VAR(QString, tf_frameid, "/ndt_frame")
    ADD_VAR(QString, tf_childframeid, "/velodyne")
    ADD_VAR(uint, bagstart, 0)
    ADD_VAR(uint, baginterval, 1)
public:
    rosbag::Bag velodynebag;
    rosbag::View * velodyneview=NULL;
    rosbag::View::const_iterator velodyneiter;
    bool tfflag;
    rosbag::Bag tfbag;
    rosbag::View * tfview=NULL;
    rosbag::View::const_iterator tfiter;
    int curframe;
public:
    ADD_VAR(u_int32_t, rosqueuesize, 1000)
    typedef ROSPub<sensor_msgs::PointCloud2> pubtype;
    ADD_INTERNAL_QOBJECT_TRIGGER(pubtype, velodynepub, 0, bagtopic_velodyne, rosqueuesize)
    ADD_INTERNAL_QOBJECT_TRIGGER(ROSTFPub, tfpub, 0, tf_childframeid, tf_frameid)
public:
    ADD_QLAYOUT(QVBoxLayout, layout)
    ADD_QWIDGET(QLabel, frame, "frameid")
    ADD_INTERNAL_QWIDGET_TRIGGER(QPushButton, next, "Next")
    ADD_INTERNAL_DEFAULT_CONNECTION(next,clicked)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{

};

//=================================================
//You can declare functions here


//=================================================
}

#endif
