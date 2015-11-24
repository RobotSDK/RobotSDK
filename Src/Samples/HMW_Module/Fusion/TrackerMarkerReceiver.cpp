#include"TrackerMarkerReceiver.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;

    if(vars->trackersub==NULL)
    {
        return 0;
    }
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
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
            fs["VelodyneExtrinsicMat"]>>vars->extrinsicmat;
            fs.release();
        }
        else
        {
            vars->extrinsicmat=cv::Mat::eye(4,4,CV_64F);
        }
    }
    else
    {
        vars->extrinsicmat=cv::Mat::eye(4,4,CV_64F);
    }

    if(vars->trackersub==NULL)
    {
        return 0;
    }
    vars->trackersub->resetTopic(vars->topic,vars->queuesize);
    vars->trackersub->resetQueryInterval(vars->queryinterval);
    vars->trackersub->startReceiveSlot();
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->trackersub->stopReceiveSlot();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;

    outputdata->tracker=vars->trackersub->getMessage();
    if(outputdata->tracker==NULL)
    {
        return 0;
    }
    int msec=(outputdata->tracker->header.stamp.sec)%(24*60*60)*1000+(outputdata->tracker->header.stamp.nsec)/1000000;
    outputdata->timestamp=QTime::fromMSecsSinceStartOfDay(msec);
    outputdata->extrinsicmat=vars->extrinsicmat.clone();
	return 1;
}
