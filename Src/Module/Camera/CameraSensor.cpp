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
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;

    if(!params->calibfilename.isEmpty())
    {
        cv::FileStorage fs;
        fs.open(params->calibfilename.toStdString(),cv::FileStorage::READ);
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
    auto vars=NODE_VARS;

    vars->camerasub->stopReceiveSlot();
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;

    outputdata->rosimage=vars->camerasub->getMessage();
    if(outputdata->rosimage==NULL)
    {
        return 0;
    }
    int msec=(outputdata->rosimage->header.stamp.sec)%(24*60*60)*1000+(outputdata->rosimage->header.stamp.nsec)/1000000;
    outputdata->timestamp=QTime::fromMSecsSinceStartOfDay(msec);

    outputdata->extrinsicmat=vars->extrinsicmat.clone();
    outputdata->cameramat=vars->cameramat.clone();
    outputdata->distcoeff=vars->distcoeff.clone();

    void * data=(void *)(outputdata->rosimage->data.data());
    if(QString::fromStdString(outputdata->rosimage->encoding)=="rgb8")
    {
        outputdata->cvimage=cv::Mat(outputdata->rosimage->height,outputdata->rosimage->width,CV_8UC3,data);
    }
    else if(QString::fromStdString(outputdata->rosimage->encoding)=="bgr8")
    {
        cv::Mat tmpimage=cv::Mat(outputdata->rosimage->height,outputdata->rosimage->width,CV_8UC3,data);
        cv::cvtColor(tmpimage,outputdata->cvimage,CV_BGR2RGB);
    }
    else if(QString::fromStdString(outputdata->rosimage->encoding)=="mono8")
    {
        outputdata->cvimage=cv::Mat(outputdata->rosimage->height,outputdata->rosimage->width,CV_8UC1,data);
    }
    else
    {
        return 0;
    }

    return 1;
}
