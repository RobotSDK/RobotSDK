#include "simulator.h"

#define SAFEWAITING 1000

Simulator::Simulator(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QTime startTime, double simulateRate, QString qstrFuncEx)
	: Source(qstrSharedLibrary,QString("Simulator"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
	starttime=startTime;
	curtime=QTime();
	nexttime=QTime();
	simulaterate=simulateRate;
	simflag=0;
    curoutputdata=boost::shared_ptr<void>();
    nextoutputdata=boost::shared_ptr<void>();
}

void Simulator::setStartTimeSlot(QTime startTime)
{
	starttime=startTime;
}

void Simulator::setSimulateRateSlot(double simulateRate)
{
	resetTimeTrackSlot();
	simulaterate=simulateRate;
}

void Simulator::syncTimeTrackSlot()
{
	if(starttime.isNull()||!openflag)
	{
		return;
	}
	curtime=QTime();
	nexttime=QTime();
    curoutputdata=boost::shared_ptr<void>();
    nextoutputdata=boost::shared_ptr<void>();
	startcurtime=QTime();
	initialcurtime=QTime();
	stopSimulatorSlot();
	Node::closeNodeSlot();
	Node::openNodeSlot();
	boost::shared_ptr<void> outputdata;
	initializeOutputData(paramsptr.get(),varsptr.get(),outputdata);
	curflag=generateSourceData(paramsptr.get(),varsptr.get(),outputdata.get(),curoutputportindex,curtime);
	while(curflag&&!curtime.isNull()&&!starttime.isNull()&&curtime<=starttime)
	{
		curflag=generateSourceData(paramsptr.get(),varsptr.get(),outputdata.get(),curoutputportindex,curtime);
	}
	if(!curflag||curtime.isNull())
	{
		emit generateSourceDataErrorSignal();
		nodeTriggerTime(NodeTriggerError);
		return;
	}
	else
	{
		curoutputdata=outputdata;
	}
	if(curtime.isNull())
	{
		emit generateSourceDataErrorSignal();
		nodeTriggerTime(NodeTriggerError);
		return;
	}
	else
	{
		curoutputdata=outputdata;
	}
}

void Simulator::resetTimeTrackSlot()
{
	if(!openflag)
	{
		return;
	}
	if(starttime.isNull())
	{
		curtime=QTime();
		nexttime=QTime();
        curoutputdata=boost::shared_ptr<void>();
        nextoutputdata=boost::shared_ptr<void>();
		startcurtime=QTime();
		initialcurtime=QTime();
		stopSimulatorSlot();
		Node::closeNodeSlot();
		Node::openNodeSlot();
	}
	else
	{
		syncTimeTrackSlot();
	}
}

void Simulator::startSimulatorSlot()
{
	if(!openflag)
	{
		return;
	}
	simflag=1;
	if(curtime.isNull()||!nexttime.isNull())
	{
		generateSourceDataSlot();
	}
	else if(!starttime.isNull())
	{
		initialcurtime=starttime.addMSecs(-SAFEWAITING);
		startcurtime=QTime::currentTime();
		int interval=int((initialcurtime.msecsTo(curtime))*simulaterate+0.5);
		QTimer::singleShot(interval,this,SOURCESLOT);
		boost::shared_ptr<void> outputdata;
		initializeOutputData(paramsptr.get(),varsptr.get(),outputdata);
		nextflag=generateSourceData(paramsptr.get(),varsptr.get(),outputdata.get(),nextoutputportindex,nexttime);
		nextoutputdata=outputdata;
	}
}

void Simulator::stopSimulatorSlot()
{
	simflag=0;
	startcurtime=QTime();
	initialcurtime=QTime();
}

void Simulator::generateSourceDataSlot()
{
	if(openflag&&simflag)
	{
		nodeTriggerTime(NodeTriggerStart);
		if(curtime.isNull())
		{
			boost::shared_ptr<void> outputdata;
			initializeOutputData(paramsptr.get(),varsptr.get(),outputdata);
			curflag=generateSourceData(paramsptr.get(),varsptr.get(),outputdata.get(),curoutputportindex,curtime);
			while(curflag&&!curtime.isNull()&&!starttime.isNull()&&curtime<=starttime)
			{
				curflag=generateSourceData(paramsptr.get(),varsptr.get(),outputdata.get(),curoutputportindex,curtime);
			}
			if(!curflag||curtime.isNull())
			{
				emit generateSourceDataErrorSignal();
				nodeTriggerTime(NodeTriggerError);
				return;
			}
			else
			{
				curoutputdata=outputdata;
			}
			initializeOutputData(paramsptr.get(),varsptr.get(),outputdata);
			nextflag=generateSourceData(paramsptr.get(),varsptr.get(),outputdata.get(),nextoutputportindex,nexttime);
			nextoutputdata=outputdata;
		}
		if(!nextflag||nexttime.isNull())
		{			
			if(curflag)
			{
				if(curoutputportindex.size()==0)
				{
					int i,n=outputports.size();
					for(i=0;i<n;i++)
					{
						outputports[i]->outputData(paramsptr,curoutputdata);
					}
				}
				else
				{
					int i,n=curoutputportindex.size();
					for(i=0;i<n;i++)
					{
						if(curoutputportindex[i]>=0&&curoutputportindex[i]<outputports.size())
						{
							outputports[curoutputportindex[i]]->outputData(paramsptr,curoutputdata);
						}
					}
				}
				emit generateSourceDataSignal();
				nodeTriggerTime(NodeTriggerEnd);
			}
			else
			{
				emit generateSourceDataErrorSignal();
				nodeTriggerTime(NodeTriggerError);
			}
			curtime=nexttime;
			curoutputportindex=nextoutputportindex;
			curoutputdata=nextoutputdata;
		}
		else
		{
			if(startcurtime.isNull())
			{
				initialcurtime=curtime;
				startcurtime=QTime::currentTime();
			}
			QTime currenttime=QTime::currentTime();
			int correctness=startcurtime.msecsTo(currenttime)-initialcurtime.msecsTo(curtime)*simulaterate;
			int interval=int((curtime.msecsTo(nexttime))*simulaterate-correctness+0.5);
			if(interval<=0)
			{
				interval=1;
			}
			QTimer::singleShot(interval,this,SOURCESLOT);
			if(curflag)
			{
				if(curoutputportindex.size()==0)
				{
					int i,n=outputports.size();
					for(i=0;i<n;i++)
					{
						outputports[i]->outputData(paramsptr,curoutputdata);
					}
				}
				else
				{
					int i,n=curoutputportindex.size();
					for(i=0;i<n;i++)
					{
						if(curoutputportindex[i]>=0&&curoutputportindex[i]<outputports.size())
						{
							outputports[curoutputportindex[i]]->outputData(paramsptr,curoutputdata);
						}
					}
				}
				emit generateSourceDataSignal();
				nodeTriggerTime(NodeTriggerEnd);
			}
			else
			{
				emit generateSourceDataErrorSignal();
				nodeTriggerTime(NodeTriggerError);
			}
			curtime=nexttime;
			curoutputportindex=nextoutputportindex;
			curoutputdata=nextoutputdata;
			boost::shared_ptr<void> outputdata;
			initializeOutputData(paramsptr.get(),varsptr.get(),outputdata);
			nextflag=generateSourceData(paramsptr.get(),varsptr.get(),outputdata.get(),nextoutputportindex,nexttime);
			nextoutputdata=outputdata;
		}
	}
}

