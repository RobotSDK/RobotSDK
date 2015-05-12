#include"CameraVirtualScanFusion.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)
PORT_DECL(1, VirtualScanGenerator)

//=================================================
//Original node functions

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->sync);
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->sync);
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;
    bool syncflag=SYNC_START(vars->sync);
    if(!syncflag)
    {
        return 0;
    }
    auto cameradata=SYNC_DATA(vars->sync, 0);
    auto virtualscandata=SYNC_DATA(vars->sync, 1);

    outputdata->timestamp=cameradata->timestamp;
    outputdata->virtualscantimestamp=virtualscandata->timestamp;
    outputdata->minrange=params->minrange;
    outputdata->maxrange=params->maxrange;
    outputdata->cvimage=cameradata->cvimage.clone();
    outputdata->virtualscan=virtualscandata->virtualscan;
    outputdata->minheights=virtualscandata->minheights;
    outputdata->maxheights=virtualscandata->maxheights;
    outputdata->labels=virtualscandata->labels;
    outputdata->clusternum=virtualscandata->clusternum;
    outputdata->clusters=virtualscandata->clusters;

    uint pointsnum=outputdata->virtualscan.size();
    cv::Mat virtualscanpoints(pointsnum*2, 4, CV_64F);
    double PI=3.141592654;
    double density=2*PI/pointsnum;
    uint i;
    for(i=0;i<pointsnum;i++)
    {
        double theta=i*density-PI;
        virtualscanpoints.at<double>(i*2,0)=outputdata->virtualscan[i]*cos(theta);
        virtualscanpoints.at<double>(i*2,1)=outputdata->virtualscan[i]*sin(theta);
        virtualscanpoints.at<double>(i*2,2)=outputdata->minheights[i];
        virtualscanpoints.at<double>(i*2,3)=1;
        virtualscanpoints.at<double>(i*2+1,0)=outputdata->virtualscan[i]*cos(theta);
        virtualscanpoints.at<double>(i*2+1,1)=outputdata->virtualscan[i]*sin(theta);
        virtualscanpoints.at<double>(i*2+1,2)=outputdata->maxheights[i];
        virtualscanpoints.at<double>(i*2+1,3)=1;
    }
    virtualscanpoints=virtualscanpoints*(virtualscandata->extrinsicmat.t())*(cameradata->extrinsicmat.inv().t());
    cv::Size imagesize;
    imagesize.width=outputdata->cvimage.cols;
    imagesize.height=outputdata->cvimage.rows;
    QPair< QPoint, QPoint > points;
    bool flag=0;
    for(i=0;i<2*pointsnum;i++)
    {
        if(virtualscanpoints.at<double>(i,2)>=params->minrange&&virtualscanpoints.at<double>(i,2)<=params->maxrange)
        {
            double tmpx=virtualscanpoints.at<double>(i,0)/virtualscanpoints.at<double>(i,2);
            double tmpy=virtualscanpoints.at<double>(i,1)/virtualscanpoints.at<double>(i,2);
            double r2=tmpx*tmpx+tmpy*tmpy;
            double tmpdist=1+cameradata->distcoeff.at<double>(0)*r2+cameradata->distcoeff.at<double>(1)*r2*r2+cameradata->distcoeff.at<double>(4)*r2*r2*r2;

            double x=tmpx*tmpdist+2*cameradata->distcoeff.at<double>(2)*tmpx*tmpy+cameradata->distcoeff.at<double>(3)*(r2+2*tmpx*tmpx);
            double y=tmpy*tmpdist+2*cameradata->distcoeff.at<double>(3)*tmpx*tmpy+cameradata->distcoeff.at<double>(2)*(r2+2*tmpy*tmpy);
            x=cameradata->cameramat.at<double>(0,0)*x+cameradata->cameramat.at<double>(0,2);
            y=cameradata->cameramat.at<double>(1,1)*y+cameradata->cameramat.at<double>(1,2);
            if(x>=0&&x<=imagesize.width-1&&y>=0&&y<=imagesize.height-1)
            {
                int px=int(x+0.5);
                int py=int(y+0.5);
                if(i%2==0)
                {
                    flag=1;
                    points.first=QPoint(px,py);
                }
                else if(i%2==1&&flag)
                {
                    flag=0;
                    points.second=QPoint(px,py);
                    outputdata->stixel.insert(outputdata->labels[i/2],points);
                }
            }
        }
    }
    return 1;
}
