#include "drain.h"

Drain::Drain(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName)
	: Node(qstrSharedLibrary,qstrNodeType,qstrNodeClass,qstrNodeName,qstrConfigName)
{
	if(outputports.size()>0)
	{
		QMessageBox::information(NULL,QString("Node Error"),QString("%1_%2_%3: Number of output ports >0").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

DrainMono::DrainMono(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Drain(qstrSharedLibrary,qstrNodeType,qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,getMonoDrainDataSizeFptr,getMonoDrainDataSize,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,processMonoDrainDataFptr,processMonoDrainData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()!=1)
	{
		QMessageBox::information(NULL,QString("Node Error"),QString("%1_%2_%3: Number of input ports !=1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

DrainMono::DrainMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Drain(qstrSharedLibrary,QString("DrainMono"),qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,getMonoDrainDataSizeFptr,getMonoDrainDataSize,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,processMonoDrainDataFptr,processMonoDrainData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()!=1)
	{
		QMessageBox::information(NULL,QString("Node Error"),QString("%1_%2_%3: Number of input ports !=1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

void DrainMono::processDrainDataSlot()
{
	if(openflag)
	{
		nodeTriggerTime(NodeTriggerStart);
		int monodatasize;
		getMonoDrainDataSize(paramsptr.get(),varsptr.get(),monodatasize);
		inputports[0]->lock();
        QVector<boost::shared_ptr<void> > boostparams=inputports[0]->grabInputParams(monodatasize);
        QVector<boost::shared_ptr<void> > boostdata=inputports[0]->grabInputData(monodatasize);
		if(monodatasize<0)
		{
			if(boostparams.size()!=-monodatasize||boostdata.size()!=-monodatasize)
			{
				inputports[0]->unlock();
				emit processDrainDataErrorSignal();
				nodeTriggerTime(NodeTriggerError);
				return;
			}
			else// if(inputports[0]->getInputBufferSize()<=0)
			{
				inputports[0]->removeInputParamsData(monodatasize);
			}
		}
		inputports[0]->unlock();
		if(processMonoDrainData(paramsptr.get(),varsptr.get(),convertBoostData(boostparams),convertBoostData(boostdata)))
		{
			emit processDrainDataSignal();
			nodeTriggerTime(NodeTriggerEnd);
		}
		else
		{
			emit processDrainDataErrorSignal();
			nodeTriggerTime(NodeTriggerError);
		}
	}
}

DrainMulti::DrainMulti(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Drain(qstrSharedLibrary,qstrNodeType,qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,getMultiDrainDataSizeFptr,getMultiDrainDataSize,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,processMultiDrainDataFptr,processMultiDrainData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()<=1)
	{
		QMessageBox::information(NULL,QString("DrainMulti Node Error"),QString("DrainMulti %1_%2_%3: Number of input ports <=1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

DrainMulti::DrainMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Drain(qstrSharedLibrary,QString("DrainMulti"),qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,getMultiDrainDataSizeFptr,getMultiDrainDataSize,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,processMultiDrainDataFptr,processMultiDrainData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()<=1)
	{
		QMessageBox::information(NULL,QString("DrainMulti Node Error"),QString("DrainMulti %1_%2_%3: Number of input ports <=1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

void DrainMulti::processDrainDataSlot()
{
	if(openflag)
	{
		nodeTriggerTime(NodeTriggerStart);
		
		int i,n=inputports.size();
        QVector<QVector<boost::shared_ptr<void> > > boostparams(n);
        QVector<QVector<void *> > drainparams(n);
        QVector<QVector<boost::shared_ptr<void> > > boostdata(n);
        QVector<QVector<void *> > draindata(n);
		QList<int> multidatasize;
		getMultiDrainDataSize(paramsptr.get(),varsptr.get(),multidatasize);
		int m=multidatasize.size();
		QVector<int> checkports;
		for(i=0;i<n;i++)
		{
			inputports[i]->lock();
			if(i<m)
			{
				boostparams[i]=inputports[i]->grabInputParams(multidatasize[i]);
				boostdata[i]=inputports[i]->grabInputData(multidatasize[i]);
				if(multidatasize[i]<0)
				{
					if(boostparams[i].size()!=-multidatasize[i]||boostdata[i].size()!=-multidatasize[i])
					{
						inputports[i]->unlock();
						emit processDrainDataErrorSignal();
						nodeTriggerTime(NodeTriggerError);
						return;
					}
					else// if(inputports[i]->getInputBufferSize()<=0)
					{
						checkports.push_back(i);
					}
				}
			}
			else
			{
				boostparams[i]=inputports[i]->grabInputParams(-1);
				boostdata[i]=inputports[i]->grabInputData(-1);
				inputports[i]->removeInputParamsData(-1);
			}
			inputports[i]->unlock();
			drainparams[i]=convertBoostData(boostparams[i]);
			draindata[i]=convertBoostData(boostdata[i]);
		}
		if(checkports.size()>0)
		{
			m=checkports.size();
			for(i=0;i<m;i++)
			{
				inputports[checkports[i]]->lock();
				inputports[checkports[i]]->removeInputParamsData(multidatasize[checkports[i]]);
				inputports[checkports[i]]->unlock();
			}
		}
		if(processMultiDrainData(paramsptr.get(),varsptr.get(),drainparams,draindata))
		{
			emit processDrainDataSignal();
			nodeTriggerTime(NodeTriggerEnd);
		}
		else
		{
			emit processDrainDataErrorSignal();
			nodeTriggerTime(NodeTriggerError);
		}
	}
}
