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
    if(vars->virtualscanpub==NULL)
    {
        return 0;
    }
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto inputdata=PORT_DATA(0,0);

    int beamnum=inputdata->virtualscan.size();

    sensor_msgs::PointCloud2 msg;
    msg.header.frame_id="VirtualScan";
    msg.header.seq=vars->seq++;
    msg.header.stamp.sec=inputdata->timestamp.second();
    msg.header.stamp.nsec=inputdata->timestamp.msec()*1000000;
    msg.height=2;
    msg.width=beamnum;
    msg.point_step=8*sizeof(float);
    msg.row_step=msg.width*msg.point_step;
    msg.data.resize(msg.height*msg.width*msg.point_step);
    unsigned char * base=msg.data.data();

    double PI=3.141592654;
    double density=2*PI/beamnum;
    int i;
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
    vars->virtualscanpub->sendMessage(msg);
    return 1;
}
