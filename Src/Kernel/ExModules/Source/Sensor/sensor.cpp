#include "sensor.h"

SensorTimer::SensorTimer(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, int msec, QString qstrFuncEx)
	: Source(qstrSharedLibrary,QString("SensorTimer"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
	timer.moveToThread(&timerthread);
	timerthread.start();
	if(msec>0)
	{
		timer.setInterval(msec);
	}
	bool flag=1;
    flag&=bool(connect(this,SIGNAL(startTimerSignal()),&timer,SLOT(start())));
    flag&=bool(connect(this,SIGNAL(startTimerSignal(int)),&timer,SLOT(start(int))));
    flag&=bool(connect(this,SIGNAL(stopTimerSignal()),&timer,SLOT(stop())));
	flag&=connectExternalTrigger(&timer,SIGNAL(timeout()),SOURCESLOT);
}

SensorTimer::~SensorTimer()
{	
	if(inputportsthread.isRunning())
	{
		emit stopTimerSignal();
		inputportsthread.exit();
		inputportsthread.wait();
	}
    bool flag=1;
    flag&=bool(disconnect(this,SIGNAL(startTimerSignal()),&timer,SLOT(start())));
    flag&=bool(disconnect(this,SIGNAL(startTimerSignal(int)),&timer,SLOT(start(int))));
    flag&=bool(disconnect(this,SIGNAL(stopTimerSignal()),&timer,SLOT(stop())));
    flag&=disconnectExternalTrigger(&timer,SIGNAL(timeout()),SOURCESLOT);
}

void SensorTimer::setTimerSlot(int msec)
{
	if(msec<=0)
	{
		return;
	}
	if(timer.isActive())
	{
		emit startTimerSignal(msec);
	}
	else
	{
		timer.setInterval(msec);
	}
}

void SensorTimer::startTimerSlot()
{
	emit startTimerSignal();
}

void SensorTimer::stopTimerSlot()
{
	emit stopTimerSignal();
}

void SensorTimer::closeNodeSlot()
{
	stopTimerSlot();
	//if(timerthread.isRunning())
	//{
	//	timerthread.exit();
	//	timerthread.wait();
	//}
	Node::closeNodeSlot();
}

SensorExternalEvent::SensorExternalEvent(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QObject * externalTrigger, const char * externalTriggerSignal, QString qstrFuncEx)
	: Source(qstrSharedLibrary,QString("SensorExternalEvent"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
	connectExternalTrigger(externalTrigger,externalTriggerSignal,SOURCESLOT);
}

SensorInternalEvent::SensorInternalEvent(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Source(qstrSharedLibrary,QString("SensorInternalEvent"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
	connectInternalTrigger(SOURCESLOT);
}
