#include"ImageTrackerMarkerFusion.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)
PORT_DECL(1, TrackerMarkerReceiver)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->sync);
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->sync);
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    bool syncflag=SYNC_START(vars->sync);
    if(!syncflag)
    {
        return 0;
    }
    auto outputdata=NODE_DATA;
    auto cameradata=SYNC_DATA(vars->sync, 0);
    auto trackerdata=SYNC_DATA(vars->sync, 1);
    outputdata->timestamp=cameradata->timestamp;
    outputdata->trackertimestamp=trackerdata->timestamp;
    int cornernum=trackerdata->tracker->points.size();
    cv::Mat cornerpoints(cornernum,4,CV_64F);
    for(int i=0;i<cornernum;i++)
    {
        cornerpoints.at<double>(i,0)=trackerdata->tracker->points[i].x;
        cornerpoints.at<double>(i,1)=trackerdata->tracker->points[i].y;
        cornerpoints.at<double>(i,2)=trackerdata->tracker->points[i].z;
        cornerpoints.at<double>(i,3)=1;
    }
    cv::Mat camerapoints=cornerpoints*(trackerdata->extrinsicmat.t())*(cameradata->extrinsicmat.inv().t());
    outputdata->cvimage=cameradata->cvimage.clone();
    cv::Size imagesize;
    imagesize.width=outputdata->cvimage.cols;
    imagesize.height=outputdata->cvimage.rows;
    for(int i=0;i<cornernum/2;i++)
    {
        if(camerapoints.at<double>(2*i,2)>1&&camerapoints.at<double>(2*i+1,2)>1)
        {
            {
                double tmpx=camerapoints.at<double>(2*i,0)/camerapoints.at<double>(2*i,2);
                double tmpy=camerapoints.at<double>(2*i,1)/camerapoints.at<double>(2*i,2);
                double r2=tmpx*tmpx+tmpy*tmpy;
                double tmpdist=1+cameradata->distcoeff.at<double>(0)*r2+cameradata->distcoeff.at<double>(1)*r2*r2+cameradata->distcoeff.at<double>(4)*r2*r2*r2;

                double x=tmpx*tmpdist+2*cameradata->distcoeff.at<double>(2)*tmpx*tmpy+cameradata->distcoeff.at<double>(3)*(r2+2*tmpx*tmpx);
                double y=tmpy*tmpdist+2*cameradata->distcoeff.at<double>(3)*tmpx*tmpy+cameradata->distcoeff.at<double>(2)*(r2+2*tmpy*tmpy);
                x=cameradata->cameramat.at<double>(0,0)*x+cameradata->cameramat.at<double>(0,2);
                y=cameradata->cameramat.at<double>(1,1)*y+cameradata->cameramat.at<double>(1,2);
                outputdata->corners.push_back(QPointF(x,y));
            }
            {
                double tmpx=camerapoints.at<double>(2*i+1,0)/camerapoints.at<double>(2*i+1,2);
                double tmpy=camerapoints.at<double>(2*i+1,1)/camerapoints.at<double>(2*i+1,2);
                double r2=tmpx*tmpx+tmpy*tmpy;
                double tmpdist=1+cameradata->distcoeff.at<double>(0)*r2+cameradata->distcoeff.at<double>(1)*r2*r2+cameradata->distcoeff.at<double>(4)*r2*r2*r2;

                double x=tmpx*tmpdist+2*cameradata->distcoeff.at<double>(2)*tmpx*tmpy+cameradata->distcoeff.at<double>(3)*(r2+2*tmpx*tmpx);
                double y=tmpy*tmpdist+2*cameradata->distcoeff.at<double>(3)*tmpx*tmpy+cameradata->distcoeff.at<double>(2)*(r2+2*tmpy*tmpy);
                x=cameradata->cameramat.at<double>(0,0)*x+cameradata->cameramat.at<double>(0,2);
                y=cameradata->cameramat.at<double>(1,1)*y+cameradata->cameramat.at<double>(1,2);
                outputdata->corners.push_back(QPointF(x,y));
            }
        }
    }
    return 1;
}
