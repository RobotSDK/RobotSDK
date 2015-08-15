#include"VirtualScanPublisher.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VirtualScanGenerator)

//=================================================
//Original node functions

//If you don't need initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(vars->virtualscanpubpointcloud2==NULL||vars->virtualscanpublaserscan==NULL)
    {
        return 0;
    }
    return 1;
}

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(vars->virtualscanpubpointcloud2==NULL||vars->virtualscanpublaserscan==NULL)
    {
        return 0;
    }
    vars->virtualscanpubpointcloud2->resetTopic(vars->topicpointcloud2,vars->queuesize);
    vars->virtualscanpublaserscan->resetTopic(vars->topiclaserscan,vars->queuesize);
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto inputdata=PORT_DATA(0,0);

    int i,beamnum=inputdata->virtualscan.size();
    double PI=3.141592654;
    double density=2*PI/beamnum;

    {
        sensor_msgs::LaserScan msg;
        msg.header.frame_id=inputdata->rospoints->header.frame_id;
        msg.header.seq=inputdata->rospoints->header.seq;
        msg.header.stamp=inputdata->rospoints->header.stamp;
        msg.angle_min=-PI;
        msg.angle_max=PI;
        msg.angle_increment=density;
        msg.time_increment=0;
        msg.scan_time=0.1;
        msg.range_min=0.1;
        msg.range_max=100;
        msg.ranges.resize(beamnum);
        msg.intensities.resize(beamnum);
        for(i=0;i<beamnum;i++)
        {
            msg.ranges[i]=inputdata->virtualscan[i];
            msg.intensities[i]=255;
        }
        vars->virtualscanpublaserscan->sendMessage(msg);
    }

    if(vars->pub3dflag)
    {
        sensor_msgs::PointCloud2 msg;
        msg.header.frame_id=inputdata->rospoints->header.frame_id;
        msg.header.seq=inputdata->rospoints->header.seq;
        msg.header.stamp=inputdata->rospoints->header.stamp;
        msg.height=2;
        msg.width=beamnum;
        msg.point_step=8*sizeof(float);
        msg.row_step=msg.width*msg.point_step;
        msg.data.resize(msg.height*msg.width*msg.point_step);
        unsigned char * base=(unsigned char *)&(msg.data[0]);

        for(i=0;i<beamnum;i++)
        {
            double theta=i*density-PI;
            float * data;
            int * ring;

            data=(float *)(base+(2*i)*msg.point_step);
            data[0]=inputdata->virtualscan[i]*cos(theta);
            data[1]=inputdata->virtualscan[i]*sin(theta);
            data[2]=inputdata->minheights[i];
            data[4]=255;
            ring=(int *)(data+5*sizeof(float));
            *ring=0;

            data=(float *)(base+(2*i+1)*msg.point_step);
            data[0]=inputdata->virtualscan[i]*cos(theta);
            data[1]=inputdata->virtualscan[i]*sin(theta);
            data[2]=inputdata->maxheights[i];
            data[4]=255;
            ring=(int *)(data+5*sizeof(float));
            *ring=1;
        }
        vars->virtualscanpubpointcloud2->sendMessage(msg);
    }
    return 1;
}
