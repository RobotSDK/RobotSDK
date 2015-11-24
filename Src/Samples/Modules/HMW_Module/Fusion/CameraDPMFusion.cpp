#include"CameraDPMFusion.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)
PORT_DECL(1, DPMDetector)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->dpmsync);
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->dpmsync);
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    bool flag=SYNC_START(vars->dpmsync);
    if(flag)
    {
        auto imagedata=SYNC_DATA(vars->dpmsync,0);
        auto dpmdata=SYNC_DATA(vars->dpmsync,1);
        auto outputdata=NODE_DATA;

        outputdata->timestamp=imagedata->timestamp;

        outputdata->cvimage=imagedata->cvimage.clone();

        outputdata->extrinsicmat=imagedata->extrinsicmat.clone();
        outputdata->cameramat=imagedata->cameramat.clone();
        outputdata->distcoeff=imagedata->distcoeff.clone();

        uint i,n=dpmdata->detection.size();
        if(n>0)
        {
            cv::Mat corners(3,n*2,CV_64F);
            for(i=0;i<n;i++)
            {
                corners.at<double>(0,i*2)=dpmdata->detection[i].x;
                corners.at<double>(1,i*2)=dpmdata->detection[i].y;
                corners.at<double>(2,i*2)=1;
                corners.at<double>(0,i*2+1)=dpmdata->detection[i].x+dpmdata->detection[i].width;
                corners.at<double>(1,i*2+1)=dpmdata->detection[i].y+dpmdata->detection[i].height;
                corners.at<double>(2,i*2+1)=1;
            }
            cv::Point2f center(imagedata->originalsize.width/2,imagedata->originalsize.height/2);
            cv::Mat rotmat=cv::getRotationMatrix2D(center,imagedata->rotation,imagedata->scale);
            corners=rotmat*corners;
            int xoffset=(imagedata->cvimage.cols-imagedata->originalsize.width)/2;
            int yoffset=(imagedata->cvimage.rows-imagedata->originalsize.height)/2;
            outputdata->detection.resize(n);
            for(i=0;i<n;i++)
            {
                if(corners.at<double>(0,i*2)<corners.at<double>(0,i*2+1))
                {
                    outputdata->detection[i].x=corners.at<double>(0,i*2)+xoffset;
                    outputdata->detection[i].width=corners.at<double>(0,i*2+1)-corners.at<double>(0,i*2);
                }
                else
                {
                    outputdata->detection[i].x=corners.at<double>(0,i*2+1)+xoffset;
                    outputdata->detection[i].width=corners.at<double>(0,i*2)-corners.at<double>(0,i*2+1);
                }
                if(corners.at<double>(1,i*2)<corners.at<double>(1,i*2+1))
                {
                    outputdata->detection[i].y=corners.at<double>(1,i*2)+yoffset;
                    outputdata->detection[i].height=corners.at<double>(1,i*2+1)-corners.at<double>(1,i*2);
                }
                else
                {
                    outputdata->detection[i].y=corners.at<double>(1,i*2+1)+yoffset;
                    outputdata->detection[i].height=corners.at<double>(1,i*2)-corners.at<double>(1,i*2+1);
                }
            }
        }
        return 1;
    }
    else
    {
        return 0;
    }
}
