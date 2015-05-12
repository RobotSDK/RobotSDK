#include"ImageProcessor.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)

//=================================================
//Original node functions

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;
    auto data=PORT_DATA(0,0);

    outputdata->timestamp=data->timestamp;

    outputdata->cvimage=data->cvimage.clone();
    cv::Point2f center(outputdata->cvimage.cols/2, outputdata->cvimage.rows/2);
    cv::Mat rotmat=cv::getRotationMatrix2D(center,vars->rotation,vars->scale);
    cv::warpAffine(outputdata->cvimage,outputdata->cvimage,rotmat,outputdata->cvimage.size());
    cv::getRectSubPix(outputdata->cvimage,cv::Size(outputdata->cvimage.cols*vars->scale,outputdata->cvimage.rows*vars->scale)
                      ,cv::Point2f(outputdata->cvimage.cols/2,outputdata->cvimage.rows/2),outputdata->cvimage);
    outputdata->cvimage.convertTo(outputdata->cvimage,-1,vars->alpha,vars->beta);

    outputdata->extrinsicmat=data->extrinsicmat.clone();
    outputdata->cameramat=data->cameramat.clone();
    outputdata->distcoeff=data->distcoeff.clone();
    double pi=3.141592654;
    double c=cos(vars->rotation*pi/180.0);
    double s=sin(vars->rotation*pi/180.0);
    cv::Mat exrotmat=cv::Mat::eye(4,4,CV_64F);
    exrotmat.at<double>(0,0)=c;exrotmat.at<double>(0,1)=-s;
    exrotmat.at<double>(1,0)=s;exrotmat.at<double>(1,1)=c;
    outputdata->extrinsicmat=outputdata->extrinsicmat*exrotmat;
    outputdata->cameramat.at<double>(0,0)*=vars->scale;outputdata->cameramat.at<double>(0,2)*=vars->scale;
    outputdata->cameramat.at<double>(1,1)*=vars->scale;outputdata->cameramat.at<double>(1,2)*=vars->scale;

    outputdata->rotation=data->rotation+vars->rotation;
    outputdata->scale=data->scale*vars->scale;

    return 1;
}

//=================================================
//Extended node functions ( rotation )

//As an extended main function, if you delete this code segment, original main function will be used
NODE_EXFUNC_DEF_EXPORT(bool, main, rotation)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;
    auto data=PORT_DATA(0,0);

    outputdata->timestamp=data->timestamp;

    outputdata->cvimage=data->cvimage.clone();
    cv::Point2f center(outputdata->cvimage.cols/2, outputdata->cvimage.rows/2);
    cv::Mat rotmat=cv::getRotationMatrix2D(center,vars->rotation,1);
    cv::warpAffine(outputdata->cvimage,outputdata->cvimage,rotmat,outputdata->cvimage.size());
    cv::getRectSubPix(outputdata->cvimage,cv::Size(outputdata->cvimage.cols*vars->scale,outputdata->cvimage.rows*vars->scale)
                      ,cv::Point2f(outputdata->cvimage.cols/2,outputdata->cvimage.rows/2),outputdata->cvimage);

    outputdata->extrinsicmat=data->extrinsicmat.clone();
    outputdata->cameramat=data->cameramat.clone();
    outputdata->distcoeff=data->distcoeff.clone();
    double pi=3.141592654;
    double c=cos(vars->rotation*pi/180.0);
    double s=sin(vars->rotation*pi/180.0);
    cv::Mat exrotmat=cv::Mat::eye(4,4,CV_64F);
    exrotmat.at<double>(0,0)=c;exrotmat.at<double>(0,1)=-s;
    exrotmat.at<double>(1,0)=s;exrotmat.at<double>(1,1)=c;
    outputdata->extrinsicmat=outputdata->extrinsicmat*exrotmat;

    outputdata->rotation=data->rotation+vars->rotation;
    outputdata->scale=data->scale;

    return 1;
}

//=================================================
//Extended node functions ( scale )

//As an extended main function, if you delete this code segment, original main function will be used
NODE_EXFUNC_DEF_EXPORT(bool, main, scale)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;
    auto data=PORT_DATA(0,0);

    outputdata->timestamp=data->timestamp;

    outputdata->cvimage=data->cvimage.clone();
    cv::Point2f center(outputdata->cvimage.cols/2, outputdata->cvimage.rows/2);
    cv::Mat rotmat=cv::getRotationMatrix2D(center,0,vars->scale);
    cv::warpAffine(outputdata->cvimage,outputdata->cvimage,rotmat,outputdata->cvimage.size());
    cv::getRectSubPix(outputdata->cvimage,cv::Size(outputdata->cvimage.cols*vars->scale,outputdata->cvimage.rows*vars->scale)
                      ,cv::Point2f(outputdata->cvimage.cols/2,outputdata->cvimage.rows/2),outputdata->cvimage);

    outputdata->extrinsicmat=data->extrinsicmat.clone();
    outputdata->cameramat=data->cameramat.clone();
    outputdata->distcoeff=data->distcoeff.clone();
    outputdata->cameramat.at<double>(0,0)*=vars->scale;outputdata->cameramat.at<double>(0,2)*=vars->scale;
    outputdata->cameramat.at<double>(1,1)*=vars->scale;outputdata->cameramat.at<double>(1,2)*=vars->scale;

    outputdata->rotation=data->rotation;
    outputdata->scale=data->scale*vars->scale;

    return 1;
}

//=================================================
//Extended node functions ( enhance )

//As an extended main function, if you delete this code segment, original main function will be used
NODE_EXFUNC_DEF_EXPORT(bool, main, enhance)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;
    auto data=PORT_DATA(0,0);

    outputdata->timestamp=data->timestamp;

    outputdata->cvimage=data->cvimage.clone();
    outputdata->cvimage.convertTo(outputdata->cvimage,-1,vars->alpha,vars->beta);

    outputdata->extrinsicmat=data->extrinsicmat.clone();
    outputdata->cameramat=data->cameramat.clone();
    outputdata->distcoeff=data->distcoeff.clone();

    outputdata->rotation=data->rotation;
    outputdata->scale=data->scale;

    return 1;
}

