#include "ObjectDetectionWidget.h"

using namespace RobotSDK_Module;

const int TrackedObjectParticlePack::statenum=TrackedStateNum;
int TrackedObjectParticlePack::particlenum=1000;
TrackedState TrackedObjectParticlePack::min={0,0,0,0,0};
TrackedState TrackedObjectParticlePack::max={0,0,0,0,0};
TrackedState TrackedObjectParticlePack::sigma={0,0,0,0,0};

TrackedObjectParticlePack::TrackedObjectParticlePack()
{
    timestamp=QTime();
    egotransform=cv::Mat::eye(4,4,CV_64F);
    ids.clear();
    deltamsec.clear();
    objects.clear();
}

void TrackedObjectParticlePack::addObject(QTime timeStamp, cv::Mat egoTransform, int id, TrackedObject object)
{
    if(!ids.contains(id))
    {
        int delta;
        if(timestamp.isNull())
        {
            timestamp=timeStamp;
            egotransform=egoTransform;
            delta=0;
        }
        else if(timestamp<timeStamp)
        {
            int i,n=objects.size();
            for(i=0;i<n;i++)
            {

            }
        }
        else if(timestamp>timeStamp)
        {

        }
        ids.push_back(id);
        deltamsec.push_back(delta);
        objects.push_back(object);
    }
}

void TrackedObject::updateEgoTransform(QTime timeStamp, cv::Mat egoTransform)
{
    if(timestamp.isNull())
    {
        timestamp=timeStamp;
        egotransform=egoTransform;
    }
    else if(timeStamp>timestamp)
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

