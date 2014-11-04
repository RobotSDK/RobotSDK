#include "processor.h"

Processor::Processor(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName)
	: Node(qstrSharedLibrary,qstrNodeType,qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,initializeOutputDataFptr,initializeOutputData,nodetype,nodeclass);
	if(outputports.size()<1)
	{
		QMessageBox::information(NULL,QString("ProcessorMono Node Error"),QString("ProcessorMono %1_%2_%3: Number of output ports <1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

ProcessorMono::ProcessorMono(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Processor(qstrSharedLibrary,qstrNodeType,qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,getMonoInputDataSizeFptr,getMonoInputDataSize,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,processMonoInputDataFptr,processMonoInputData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()!=1)
	{
		QMessageBox::information(NULL,QString("ProcessorMono Node Error"),QString("ProcessorMono %1_%2_%3: Number of input ports !=1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

ProcessorMono::ProcessorMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Processor(qstrSharedLibrary,QString("ProcessorMono"),qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,getMonoInputDataSizeFptr,getMonoInputDataSize,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,processMonoInputDataFptr,processMonoInputData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()!=1)
	{
		QMessageBox::information(NULL,QString("ProcessorMono Node Error"),QString("ProcessorMono %1_%2_%3: Number of input ports !=1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

void ProcessorMono::processInputDataSlot()
{
	if(openflag)
	{
		nodeTriggerTime(NodeTriggerStart);		
		int monodatasize;
		getMonoInputDataSize(paramsptr.get(),varsptr.get(),monodatasize);
		inputports[0]->lock();
        QVector<boost::shared_ptr<void> > boostparams=inputports[0]->grabInputParams(monodatasize);
        QVector<boost::shared_ptr<void> > boostdata=inputports[0]->grabInputData(monodatasize);
		if(monodatasize<0)
		{
			if(boostparams.size()!=-monodatasize||boostdata.size()!=-monodatasize)
			{
				inputports[0]->unlock();
				emit processInputDataErrorSignal();
				nodeTriggerTime(NodeTriggerError);
				return;
			}
			else// if(inputports[0]->getInputBufferSize()<=0)
			{
				inputports[0]->removeInputParamsData(monodatasize);
			}
		}
		inputports[0]->unlock();
		boost::shared_ptr<void> outputdata;
		initializeOutputData(paramsptr.get(),varsptr.get(),outputdata);
		QList<int> outputportindex;
		if(processMonoInputData(paramsptr.get(),varsptr.get(),convertBoostData(boostparams),convertBoostData(boostdata),outputdata.get(),outputportindex))
		{
			if(outputportindex.size()==0)
			{
				int i,n=outputports.size();
				for(i=0;i<n;i++)
				{
					outputports[i]->outputData(paramsptr,outputdata);
				}
			}
			else
			{
				int i,n=outputportindex.size();
				for(i=0;i<n;i++)
				{
					if(outputportindex[i]>=0&&outputportindex[i]<outputports.size())
					{
						outputports[outputportindex[i]]->outputData(paramsptr,outputdata);
					}
				}
			}
			emit processInputDataSignal();
			nodeTriggerTime(NodeTriggerEnd);
		}
		else
		{
			emit processInputDataErrorSignal();
			nodeTriggerTime(NodeTriggerError);
		}
	}
}

ProcessorMulti::ProcessorMulti(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Processor(qstrSharedLibrary,qstrNodeType,qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,getMultiInputDataSizeFptr,getMultiInputDataSize,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,processMultiInputDataFptr,processMultiInputData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()<=1)
	{
		QMessageBox::information(NULL,QString("ProcessorMulti Node Error"),QString("ProcessorMulti %1_%2_%3: Number of input ports <=1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

ProcessorMulti::ProcessorMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Processor(qstrSharedLibrary,QString("ProcessorMulti"),qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,getMultiInputDataSizeFptr,getMultiInputDataSize,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,processMultiInputDataFptr,processMultiInputData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()<=1)
	{
		QMessageBox::information(NULL,QString("ProcessorMulti Node Error"),QString("ProcessorMulti %1_%2_%3: Number of input ports <=1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

void ProcessorMulti::processInputDataSlot()
{
	if(openflag)
	{
		nodeTriggerTime(NodeTriggerStart);
		int i,n=inputports.size();
        QVector<QVector<boost::shared_ptr<void> > > boostparams(n);
        QVector<QVector<boost::shared_ptr<void> > > boostdata(n);
        QVector<QVector<void *> > inputparams(n);
        QVector<QVector<void *> > inputdata(n);
		QList<int> multidatasize;
		getMultiInputDataSize(paramsptr.get(),varsptr.get(),multidatasize);
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
						emit processInputDataErrorSignal();
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
			inputparams[i]=convertBoostData(boostparams[i]);
			inputdata[i]=convertBoostData(boostdata[i]);
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
		boost::shared_ptr<void> outputdata;
		initializeOutputData(paramsptr.get(),varsptr.get(),outputdata);
		QList<int> outputportindex;
		if(processMultiInputData(paramsptr.get(),varsptr.get(),inputparams,inputdata,outputdata.get(),outputportindex))
		{
			if(outputportindex.size()==0)
			{
				int i,n=outputports.size();
				for(i=0;i<n;i++)
				{
					outputports[i]->outputData(paramsptr,outputdata);
				}
			}
			else
			{
				int i,n=outputportindex.size();
				for(i=0;i<n;i++)
				{
					if(outputportindex[i]>=0&&outputportindex[i]<outputports.size())
					{
						outputports[outputportindex[i]]->outputData(paramsptr,outputdata);
					}
				}
			}
			emit processInputDataSignal();
			nodeTriggerTime(NodeTriggerEnd);
		}
		else
		{
			emit processInputDataErrorSignal();
			nodeTriggerTime(NodeTriggerError);
		}
	}
}
