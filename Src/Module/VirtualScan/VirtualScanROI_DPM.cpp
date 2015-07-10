#include"VirtualScanROI_DPM.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VirtualScanGenerator)
PORT_DECL(1, DPMDetector)
PORT_DECL(2, CameraSensor)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->sync.clear();
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->sync.clear();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(SYNC_START(vars->sync))
    {
        auto vscan=SYNC_DATA(vars->sync,0);
        auto dpm=SYNC_DATA(vars->sync,1);
        auto image=SYNC_DATA(vars->sync,2);
        auto data=NODE_DATA;
        data->timestamp=vscan->timestamp;
        int pointsnum=vscan->virtualscan.size();
        cv::Mat virtualscanpoints(pointsnum, 4, CV_64F);
        double PI=3.141592654;
        double density=2*PI/pointsnum;

        for(int i=0;i<pointsnum;i++)
        {
            double theta=i*density-PI;
            virtualscanpoints.at<double>(i,0)=vscan->virtualscan[i]*cos(theta);
            virtualscanpoints.at<double>(i,1)=vscan->virtualscan[i]*sin(theta);
            virtualscanpoints.at<double>(i,2)=(vscan->minheights[i]+vscan->maxheights[i])/2;
            virtualscanpoints.at<double>(i,3)=1;
        }
        virtualscanpoints=virtualscanpoints*(vscan->extrinsicmat.t())*(image->extrinsicmat.inv().t());

        QVector<QPoint> imagepoints;
        imagepoints.resize(pointsnum);
        QVector<bool> pointsflag;
        pointsflag.resize(pointsnum);
        cv::Size imagesize;
        imagesize.width=image->cvimage.cols;
        imagesize.height=image->cvimage.rows;
        for(int i=0;i<pointsnum;i++)
        {
            if(virtualscanpoints.at<double>(i,2)>=vars->minrange&&virtualscanpoints.at<double>(i,2)<=vars->maxrange)
            {
                double tmpx=virtualscanpoints.at<double>(i,0)/virtualscanpoints.at<double>(i,2);
                double tmpy=virtualscanpoints.at<double>(i,1)/virtualscanpoints.at<double>(i,2);
                double r2=tmpx*tmpx+tmpy*tmpy;
                double tmpdist=1+image->distcoeff.at<double>(0)*r2+image->distcoeff.at<double>(1)*r2*r2+image->distcoeff.at<double>(4)*r2*r2*r2;

                double x=tmpx*tmpdist+2*image->distcoeff.at<double>(2)*tmpx*tmpy+image->distcoeff.at<double>(3)*(r2+2*tmpx*tmpx);
                double y=tmpy*tmpdist+2*image->distcoeff.at<double>(3)*tmpx*tmpy+image->distcoeff.at<double>(2)*(r2+2*tmpy*tmpy);
                x=image->cameramat.at<double>(0,0)*x+image->cameramat.at<double>(0,2);
                y=image->cameramat.at<double>(1,1)*y+image->cameramat.at<double>(1,2);
                if(x>=0&&x<=imagesize.width-1&&y>=0&&y<=imagesize.height-1)
                {
                    int px=int(x+0.5);
                    int py=int(y+0.5);
                    imagepoints[i]=QPoint(x,y);
                    pointsflag[i]=(1);
                }
                else
                {
                    imagepoints[i]=QPoint();
                    pointsflag[i]=(0);
                }
            }
            else
            {
                imagepoints[i]=QPoint();
                pointsflag[i]=(0);
            }
        }

        for(int i=0;i<dpm->detection.size();i++)
        {
            QRect rect(dpm->detection[i].x,dpm->detection[i].y,dpm->detection[i].width,dpm->detection[i].height);
            int start=-1;
            int end=-1;
            for(int j=0;j<pointsnum;j++)
            {
                if(pointsflag[j]&&rect.contains(imagepoints[j]))
                {
                    if(start<0)
                    {
                        start=j;
                        end=start;
                    }
                    else
                    {
                        end=j;
                    }
                }
            }
            if(start>=0)
            {
                data->roi.push_back(QPair<int,int>(start,end));
            }
        }
        return 1;
    }
    return 0;
}
