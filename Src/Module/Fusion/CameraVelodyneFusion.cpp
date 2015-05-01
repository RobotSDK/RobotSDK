#include"CameraVelodyneFusion.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)
PORT_DECL(1, VelodyneSensor)

//=================================================
//Original node functions

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->sync);
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->sync);
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;
    bool syncflag=SYNC_START(vars->sync);
    if(!syncflag)
    {
        return 0;
    }
    auto cameradata=SYNC_DATA(vars->sync, 0);
    auto velodynedata=SYNC_DATA(vars->sync, 1);
    outputdata->timestamp=cameradata->timestamp;
    outputdata->velodynetimestamp=velodynedata->timestamp;
    outputdata->minrange=params->minrange;
    outputdata->maxrange=params->maxrange;
    uint pointsnum=velodynedata->pclpoints->points.size();
    cv::Mat velodynepoints(pointsnum, 8, CV_32F, velodynedata->pclpoints->points.data());
    cv::Mat pointscolor;
    velodynepoints.convertTo(pointscolor,CV_64F);
    cv::Mat points=pointscolor(cv::Rect(0,0,4,pointsnum));
    cv::Mat camerapoints=points*(velodynedata->extrinsicmat.t())*(cameradata->extrinsicmat.inv().t());
    outputdata->cvimage=cameradata->cvimage.clone();
    cv::Size imagesize;
    imagesize.width=outputdata->cvimage.cols;
    imagesize.height=outputdata->cvimage.rows;
    uint i;
    for(i=0;i<pointsnum;i++)
    {
        if(camerapoints.at<double>(i,2)>=params->minrange&&camerapoints.at<double>(i,2)<=params->maxrange)
        {
            double tmpx=camerapoints.at<double>(i,0)/camerapoints.at<double>(i,2);
            double tmpy=camerapoints.at<double>(i,1)/camerapoints.at<double>(i,2);
            double r2=tmpx*tmpx+tmpy*tmpy;
            double tmpdist=1+cameradata->distcoeff.at<double>(0)*r2+cameradata->distcoeff.at<double>(1)*r2*r2+cameradata->distcoeff.at<double>(4)*r2*r2*r2;

            double x=tmpx*tmpdist+2*cameradata->distcoeff.at<double>(2)*tmpx*tmpy+cameradata->distcoeff.at<double>(3)*(r2+2*tmpx*tmpx);
            double y=tmpy*tmpdist+2*cameradata->distcoeff.at<double>(3)*tmpx*tmpy+cameradata->distcoeff.at<double>(2)*(r2+2*tmpy*tmpy);
            x=cameradata->cameramat.at<double>(0,0)*x+cameradata->cameramat.at<double>(0,2);
            y=cameradata->cameramat.at<double>(1,1)*y+cameradata->cameramat.at<double>(1,2);
            if(x>=0 && x<=imagesize.width-1 && y>=0 && y<=imagesize.height-1)
            {
                int px=int(x+0.5);
                int py=int(y+0.5);
                QPair<int, int> point=QPair<int, int>(px,py);
                double range=camerapoints.at<double>(i,2);
                if(!outputdata->ranges.contains(point))
                {
                    outputdata->ranges.insert(point,range);
                }
                else
                {
                    if(outputdata->ranges[point]>range)
                    {
                        outputdata->ranges[point]=range;
                    }
                }
            }
        }
    }
    return 1;
}
