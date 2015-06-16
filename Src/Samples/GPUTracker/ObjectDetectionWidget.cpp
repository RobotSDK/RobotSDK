#include "ObjectDetectionWidget.h"

using namespace RobotSDK_Module;

const int TrackedObject::statenum=TrackedStateNum;
TrackedState TrackedObject::min;
TrackedState TrackedObject::max;
TrackedState TrackedObject::sigma;

TrackedObject::TrackedObject(QTime timeStamp, cv::Mat egoTransform, TrackedState objectState)
{
    initialflag=1;
    timestamp=timeStamp;
    egotransform=egoTransform;
    objectstate=objectState;
}

void TrackedObject::updateEgoTransform(QTime timeStamp, cv::Mat egoTransform)
{
    if(timeStamp>timestamp)
    {
        cv::Mat delta=egoTransform.inv()*egotransform;
        cv::Mat position(4,1,CV_64F);
        position.at<double>(0)=objectstate.x;
        position.at<double>(1)=objectstate.y;
        position.at<double>(2)=0;
        position.at<double>(3)=1;
        position=delta*position;
        objectstate.x=position.at<double>(0);
        objectstate.y=position.at<double>(1);
        cv::Mat orientation(3,1,CV_64F);
        orientation.at<double>(0)=cos(objectstate.theta);
        orientation.at<double>(1)=sin(objectstate.theta);
        orientation.at<double>(2)=0;
        orientation=delta(cv::Rect(0,0,3,3))*orientation;
        objectstate.theta=atan2(orientation.at<double>(1),orientation.at<double>(0));
        timestamp=timeStamp;
        egotransform=egoTransform;
        initialflag=0;
    }
    return;
}

TrackedState TrackedObject::getObjectState()
{
    return objectstate;
}

void TrackedObject::setObjectState(TrackedState objectState)
{
    objectstate=objectState;
    return;
}

ObjectDetectionWidget::ObjectDetectionWidget(QWidget *parent)
    :QGraphicsView(parent)
{

}

