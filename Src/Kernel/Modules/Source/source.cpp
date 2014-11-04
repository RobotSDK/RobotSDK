#include "source.h"

Source::Source(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Node(qstrSharedLibrary,qstrNodeType,qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,initializeOutputDataFptr,initializeOutputData,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,generateSourceDataFptr,generateSourceData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()>0)
	{
		QMessageBox::information(NULL,QString("Source Node Error"),QString("Source %1_%2_%3: Number of input ports >0").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
	if(outputports.size()<1)
	{
		QMessageBox::information(NULL,QString("Source Node Error"),QString("Source %1_%2_%3: Number of output ports <1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

Source::Source(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Node(qstrSharedLibrary,QString("Source"),qstrNodeClass,qstrNodeName,qstrConfigName)
{
	LoadCheckFptr(sharedlibrary,initializeOutputDataFptr,initializeOutputData,nodetype,nodeclass);
	LoadCheckExFptr(sharedlibrary,generateSourceDataFptr,generateSourceData,qstrFuncEx,nodetype,nodeclass);
	if(inputports.size()>0)
	{
		QMessageBox::information(NULL,QString("Source Node Error"),QString("Source %1_%2_%3: Number of input ports >0").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
	if(outputports.size()<1)
	{
		QMessageBox::information(NULL,QString("Source Node Error"),QString("Source %1_%2_%3: Number of output ports <1").arg(nodetype).arg(nodeclass).arg(nodename));
		exit(0);
	}
}

void Source::generateSourceDataSlot()
{
	if(openflag)
	{
		nodeTriggerTime(NodeTriggerStart);
		boost::shared_ptr<void> outputdata;
		initializeOutputData(paramsptr.get(),varsptr.get(),outputdata);
		QList<int> outputportindex;
		QTime timestamp;
		if(generateSourceData(paramsptr.get(),varsptr.get(),outputdata.get(),outputportindex,timestamp))
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
			emit generateSourceDataSignal();
			nodeTriggerTime(NodeTriggerEnd);
		}
		else
		{
			emit generateSourceDataErrorSignal();
			nodeTriggerTime(NodeTriggerError);
		}
	}
}
