#include"NDTLocalizer.h"
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
    if(vars->ndtsub==NULL)
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

    if(vars->ndtsub==NULL)
    {
        return 0;
    }
    vars->ndtsub->resetDestinationFrame(vars->destinationframe);
    vars->ndtsub->resetOriginalFrame(vars->originalframe);
    vars->ndtsub->startReceiveSlot();
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->ndtsub->stopReceiveSlot();
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;

    tf::StampedTransform rostransform;
    if(!(vars->ndtsub->getTF(rostransform)))
    {
        return 0;
    }
    int msec=(rostransform.stamp_.sec)%(24*60*60)*1000+(rostransform.stamp_.nsec)/1000000;
    data->timestamp=QTime::fromMSecsSinceStartOfDay(msec);
    tf::Matrix3x3 rotation(rostransform.getRotation());
    tf::Vector3 translation=rostransform.getOrigin();
    uint i;
    data->cvtransform=cv::Mat::eye(4,4,CV_64F);
    for(i=0;i<3;i++)
    {
        data->cvtransform.at<double>(i,0)=(double)(rotation.getRow(i).x());
        data->cvtransform.at<double>(i,1)=(double)(rotation.getRow(i).y());
        data->cvtransform.at<double>(i,2)=(double)(rotation.getRow(i).z());
    }
    data->cvtransform.at<double>(0,3)=(double)(translation.x());
    data->cvtransform.at<double>(1,3)=(double)(translation.y());
    data->cvtransform.at<double>(2,3)=(double)(translation.z());

    data->extrinsicmat=vars->extrinsicmat.clone();
    return 1;
}
