#include"CameraSensor.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Original node functions

//If you don't need initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;

    if(vars->camerasub==NULL)
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

    if(!vars->calibfilename.isEmpty())
    {
        cv::FileStorage fs;
        fs.open(vars->calibfilename.toStdString(),cv::FileStorage::READ);
        if(fs.isOpened())
        {
            fs["CameraExtrinsicMat"]>>vars->extrinsicmat;
            fs["CameraMat"]>>vars->cameramat;
            fs["DistCoeff"]>>vars->distcoeff;
            fs.release();
        }
        else
        {
            vars->extrinsicmat=cv::Mat::eye(4,4,CV_64F);
            vars->cameramat=cv::Mat::eye(3,3,CV_64F);
            vars->distcoeff=cv::Mat::zeros(1,5,CV_64F);
        }
    }
    else
    {
        vars->extrinsicmat=cv::Mat::eye(4,4,CV_64F);
        vars->cameramat=cv::Mat::eye(3,3,CV_64F);
        vars->distcoeff=cv::Mat::zeros(1,5,CV_64F);
    }    
    if(vars->camerasub==NULL)
    {
        return 0;
    }
    vars->camerasub->resetTopic(vars->topic,vars->queuesize);
    vars->camerasub->resetQueryInterval(vars->queryinterval);
    vars->camerasub->startReceiveSlot();
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;

    vars->camerasub->stopReceiveSlot();
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;

    auto rosimage=vars->camerasub->getMessage();
    if(rosimage==NULL)
    {
        return 0;
    }    

    int msec=(rosimage->header.stamp.sec)%(24*60*60)*1000+(rosimage->header.stamp.nsec)/1000000;
    outputdata->timestamp=QTime::fromMSecsSinceStartOfDay(msec);

    outputdata->cvimage=cv_bridge::toCvShare(rosimage)->image.clone();
    if(rosimage->encoding=="bgr8")
    {
        cv::cvtColor(outputdata->cvimage,outputdata->cvimage,CV_BGR2RGB);
    }
    outputdata->originalsize.width=outputdata->cvimage.cols;
    outputdata->originalsize.height=outputdata->cvimage.rows;

    outputdata->extrinsicmat=vars->extrinsicmat.clone();
    outputdata->cameramat=vars->cameramat.clone();
    outputdata->distcoeff=vars->distcoeff.clone();   

    return 1;
}
