#include"ImageProcessor.h"

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Original node functions

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto params=NODE_PARAMS;
    auto data=PORT_DATA(0,0);
    auto outputdata=NODE_DATA;

    outputdata->timestamp=data->timestamp;

    outputdata->extrinsicmat=data->extrinsicmat;
    outputdata->cameramat=data->cameramat;
    outputdata->distcoeff=data->distcoeff;

    outputdata->rosimage=data->rosimage;

    cv::Mat image;
    cv::Point2f center(data->cvimage.cols/2, data->cvimage.rows/2);
    cv::Mat rotmat=cv::getRotationMatrix2D(center,params->angle,params->ratio);
    cv::warpAffine(data->cvimage,image,rotmat,data->cvimage.size());
    cv::getRectSubPix(image,cv::Size(image.cols*params->ratio,image.rows*params->ratio),cv::Point2f(image.cols/2,image.rows/2),image);

    double pi=3.141592654;
    double c=cos(params->angle*pi/180.0);
    double s=sin(params->angle*pi/180.0);
    cv::Mat exrotmat=cv::Mat::eye(4,4,CV_64F);
    exrotmat.at<double>(0,0)=c;exrotmat.at<double>(0,1)=s;
    exrotmat.at<double>(1,0)=-s;exrotmat.at<double>(1,1)=c;
    outputdata->extrinsicmat=outputdata->extrinsicmat*exrotmat;

    outputdata->cameramat.at<double>(0,0)*=params->ratio;outputdata->cameramat.at<double>(0,2)*=params->ratio;
    outputdata->cameramat.at<double>(1,1)*=params->ratio;outputdata->cameramat.at<double>(1,2)*=params->ratio;

    image.convertTo(image,-1,params->alpha,params->beta);

    outputdata->cvimage=image;
    return 1;
}

//=================================================
//Extended node functions ( rotation )

//As an extended main function, if you delete this code segment, original main function will be used
NODE_EXFUNC_DEF_EXPORT(bool, main, rotation)
{
    auto params=NODE_PARAMS;
    auto data=PORT_DATA(0,0);
    auto outputdata=NODE_DATA;

    outputdata->timestamp=data->timestamp;

    outputdata->extrinsicmat=data->extrinsicmat;
    outputdata->cameramat=data->cameramat;
    outputdata->distcoeff=data->distcoeff;

    outputdata->rosimage=data->rosimage;

    cv::Mat image;
    cv::Point2f center(data->cvimage.cols/2, data->cvimage.rows/2);
    cv::Mat rotmat=cv::getRotationMatrix2D(center,params->angle,1.0);
    cv::warpAffine(data->cvimage,image,rotmat,data->cvimage.size());

    double pi=3.141592654;
    double c=cos(params->angle*pi/180.0);
    double s=sin(params->angle*pi/180.0);
    cv::Mat exrotmat=cv::Mat::eye(4,4,CV_64F);
    exrotmat.at<double>(0,0)=c;exrotmat.at<double>(0,1)=s;
    exrotmat.at<double>(1,0)=-s;exrotmat.at<double>(1,1)=c;
    outputdata->extrinsicmat=outputdata->extrinsicmat*exrotmat;

    outputdata->cvimage=image;
    return 1;
}

//=================================================
//Extended node functions ( scale )

//As an extended main function, if you delete this code segment, original main function will be used
NODE_EXFUNC_DEF_EXPORT(bool, main, scale)
{
    auto params=NODE_PARAMS;
    auto data=PORT_DATA(0,0);
    auto outputdata=NODE_DATA;

    outputdata->timestamp=data->timestamp;

    outputdata->extrinsicmat=data->extrinsicmat;
    outputdata->cameramat=data->cameramat;
    outputdata->distcoeff=data->distcoeff;

    outputdata->rosimage=data->rosimage;

    cv::Mat image;
    cv::Point2f center(data->cvimage.cols/2, data->cvimage.rows/2);
    cv::Mat rotmat=cv::getRotationMatrix2D(center,0,params->ratio);
    cv::warpAffine(data->cvimage,image,rotmat,data->cvimage.size());
    cv::getRectSubPix(image,cv::Size(image.cols*params->ratio,image.rows*params->ratio),cv::Point2f(image.cols/2,image.rows/2),image);

    outputdata->cameramat.at<double>(0,0)*=params->ratio;outputdata->cameramat.at<double>(0,2)*=params->ratio;
    outputdata->cameramat.at<double>(1,1)*=params->ratio;outputdata->cameramat.at<double>(1,2)*=params->ratio;

    outputdata->cvimage=image;
    return 1;
}

//=================================================
//Extended node functions ( enhance )

//As an extended main function, if you delete this code segment, original main function will be used
NODE_EXFUNC_DEF_EXPORT(bool, main, enhance)
{
    auto params=NODE_PARAMS;
    auto data=PORT_DATA(0,0);
    auto outputdata=NODE_DATA;

    outputdata->timestamp=data->timestamp;

    outputdata->extrinsicmat=data->extrinsicmat;
    outputdata->cameramat=data->cameramat;
    outputdata->distcoeff=data->distcoeff;

    outputdata->rosimage=data->rosimage;

    cv::Mat image=data->cvimage.clone();
    image.convertTo(image,-1,params->alpha,params->beta);

    outputdata->cvimage=image;
    return 1;
}

