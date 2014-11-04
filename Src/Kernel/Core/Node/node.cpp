#include "node.h"

InputPort::InputPort(int inputBufferSize)
{
    inputbuffersize=inputBufferSize;
	paramsgrubcount=0;
	datagrubcount=0;
	checkstop=1;
}

InputPort::~InputPort()
{
    inputparamsbuffer.clear();
    inputdatabuffer.clear();
}

int InputPort::getInputBufferSize()
{
	return inputbuffersize;
}

void InputPort::lock()
{
	readwritelock.lockForRead();
}

void InputPort::unlock()
{
	readwritelock.unlock();
}

QVector<boost::shared_ptr<void> > InputPort:: grabInputParams(int grabSize)
{
    QVector<boost::shared_ptr<void> > result;
    int tmpbuffersize=inputparamsbuffer.size();
    if(tmpbuffersize==0)
    {
        return result;
    }
    if(grabSize==0)
    {
        result=QVector<boost::shared_ptr<void> >::fromList(inputparamsbuffer);
        if(inputbuffersize<=0)
        {
            inputparamsbuffer.clear();
        }
		else if(checkstop)
		{
			paramsgrubcount++;
			if(paramsgrubcount>CHECKSTOP*inputbuffersize)
			{
				inputparamsbuffer.clear();
			}
		}
    }
    else if(grabSize>0)
    {
        if(grabSize>=tmpbuffersize)
        {
            result=QVector<boost::shared_ptr<void> >::fromList(inputparamsbuffer);
        }
        else
        {
            result=QVector<boost::shared_ptr<void> >::fromList(inputparamsbuffer.mid(0,grabSize));
        }
		if(inputbuffersize>0&&checkstop)
		{
			paramsgrubcount++;
			if(paramsgrubcount>CHECKSTOP*inputbuffersize)
			{
				inputparamsbuffer.clear();
			}
		}
    }
    else
    {
        grabSize=-grabSize;
        if(grabSize<=tmpbuffersize)
        {
            result=QVector<boost::shared_ptr<void> >::fromList(inputparamsbuffer.mid(tmpbuffersize-grabSize,grabSize));
        }
    }
    return result;
}

QVector<boost::shared_ptr<void> > InputPort::grabInputData(int grabSize)
{
    QVector<boost::shared_ptr<void> > result;
    int tmpbuffersize=inputdatabuffer.size();
    if(tmpbuffersize==0)
    {
        return result;
    }
    if(grabSize==0)
    {
        result=QVector<boost::shared_ptr<void> >::fromList(inputdatabuffer);
        if(inputbuffersize<=0)
        {
            inputdatabuffer.clear();
        }
		else if(checkstop)
		{
			datagrubcount++;
			if(datagrubcount>CHECKSTOP*inputbuffersize)
			{
				inputdatabuffer.clear();
			}
		}
    }
    else if(grabSize>0)
    {
        if(grabSize>=tmpbuffersize)
        {
            result=QVector<boost::shared_ptr<void> >::fromList(inputdatabuffer);
        }
        else
        {
            result=QVector<boost::shared_ptr<void> >::fromList(inputdatabuffer.mid(0,grabSize));
        }
		if(inputbuffersize>0&&checkstop)
		{
			datagrubcount++;
			if(datagrubcount>CHECKSTOP*inputbuffersize)
			{
				inputdatabuffer.clear();
			}
		}
    }
    else
    {
        grabSize=-grabSize;
        if(grabSize<=tmpbuffersize)
        {
            result=QVector<boost::shared_ptr<void> >::fromList(inputdatabuffer.mid(tmpbuffersize-grabSize,grabSize));
        }
    }
    return result;
}

void InputPort::removeInputParamsData(int removeSize)
{	
	if(removeSize<0)//&&inputbuffersize<=0)
	{		
		removeSize=-removeSize;
		int tmpbuffersize=inputdatabuffer.size();
		if(removeSize>=tmpbuffersize)
		{
			inputparamsbuffer.clear();
			inputdatabuffer.clear();
		}
		else
		{
			inputparamsbuffer=inputparamsbuffer.mid(0,tmpbuffersize-removeSize);
			inputdatabuffer=inputdatabuffer.mid(0,tmpbuffersize-removeSize);
		}     			
	}
	return;
}

void InputPort::clear()
{
    inputparamsbuffer.clear();
    inputdatabuffer.clear();
}

void InputPort::setCheckStop(bool flag)
{
	checkstop=flag;
}

void InputPort::inputDataSlot(boost::shared_ptr<void> inputParamsPtr,  boost::shared_ptr<void> inputDataPtr)
{
    readwritelock.lockForWrite();
	paramsgrubcount=0;
	datagrubcount=0;
    inputparamsbuffer.push_front(inputParamsPtr);
    inputdatabuffer.push_front(inputDataPtr);
    int tmpbuffersize=inputdatabuffer.size();
    if(inputbuffersize>0&&tmpbuffersize>inputbuffersize)
    {
        inputparamsbuffer.pop_back();
        inputdatabuffer.pop_back();
    }
    readwritelock.unlock();
    emit inputDataSignal();
}

OutputPort::OutputPort(QObject * parent)
    : QObject(parent)
{
}

OutputPort::~OutputPort()
{
}

void OutputPort::outputData(boost::shared_ptr<void> outputParamsPtr, boost::shared_ptr<void> outputDataPtr)
{
    emit outputDataSignal(outputParamsPtr,outputDataPtr);
}

Node::Node(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName)
{ 
    QString sharedlibraryname=qstrSharedLibrary;

#ifdef _DEBUG
    sharedlibraryname=sharedlibraryname+QString("_Debug");
#else
    sharedlibraryname=sharedlibraryname+QString("_Release");
#endif

    sharedlibrary.setFileName(sharedlibraryname);
    bool loadflag=sharedlibrary.load();

 #ifdef Q_OS_LINUX
    if(!loadflag)
    {
        sharedlibrary.setFileName(QString("%1/SDK/RobotSDK/Module/SharedLibrary/%2").arg(QString(qgetenv("HOME"))).arg(sharedlibraryname));
        loadflag=sharedlibrary.load();
    }
#elif defined(Q_OS_WIN)

#endif

    nodetype=qstrNodeType;
    nodeclass=qstrNodeClass;
    nodename=qstrNodeName;
    configname=qstrConfigName;

    if(loadflag)
    {
        LoadCheckFptr(sharedlibrary,getPortsSizeFptr,getPortsSize,nodetype,nodeclass);
        LoadCheckFptr(sharedlibrary,initializeParamsFptr,initializeParams,nodetype,nodeclass);
        LoadCheckFptr(sharedlibrary,initializeVarsFptr,initializeVars,nodetype,nodeclass);
        LoadCheckFptr(sharedlibrary,setParamsVarsOpenNodeFptr,setParamsVarsOpenNode,nodetype,nodeclass);
        LoadCheckFptr(sharedlibrary,handleVarsCloseNodeFptr,handleVarsCloseNode,nodetype,nodeclass);
        LoadCheckFptr(sharedlibrary,getInternalTriggerFptr,getInternalTrigger,nodetype,nodeclass);
    }
    else
    {
        QMessageBox::information(NULL,QString("Node Interface Function Load Error"),sharedlibrary.errorString());
        exit(0);
    }
    openflag=0;

    initializeParams(paramsptr);
    initializeVars(varsptr);

    QList<int> inputportssize;
    int outputportsnumber;
    getPortsSize(inputportssize,outputportsnumber);
    if(inputportssize.size()>0)
    {
        int i,n=inputportssize.size();
        inputports.resize(n);
        for(i=0;i<n;i++)
        {
            inputports[i]=new InputPort(inputportssize.at(i));
            inputports[i]->moveToThread(&inputportsthread);
        }
        inputportsthread.start();
    }
    if(outputportsnumber>0)
    {
        int i,n=outputportsnumber;
        outputports.resize(n);
        for(i=0;i<n;i++)
        {
            outputports[i]=new OutputPort(this);
        }
    }
}

Node::~Node()
{
    if(openflag)
    {
        closeNodeSlot();
    }
    if(inputportsthread.isRunning())
    {
        inputportsthread.exit();
        inputportsthread.wait();
    }
    if(inputports.size()>0)
    {
        int i,n=inputports.size();
        for(i=0;i<n;i++)
        {
            inputports[i]->deleteLater();
        }
        inputports.clear();
    }
}

QString Node::getSharedLibraryName()
{
    return sharedlibrary.fileName();
}

QString Node::getNodeType()
{

    return nodetype;
}

QString Node::getNodeClass()
{
    return nodeclass;
}

QString Node::getNodeName()
{
    return nodename;
}

QString Node::getConfigName()
{
    return configname;
}

void * Node::getParamsPtr()
{
    return paramsptr.get();
}

void * Node::getVarsPtr()
{
    return paramsptr.get();
}

bool Node::connectExternalTrigger(QObject * externalTrigger, const char * externalTriggerSignal, const char * triggerSlot)
{
    return bool(connect(externalTrigger,externalTriggerSignal,this,triggerSlot));
}

bool Node::connectExternalTrigger(int inputPortIndex, const char * triggerSlot)
{
    if(inputPortIndex>=0&&inputPortIndex<inputports.size())
    {
        return bool(connect(inputports[inputPortIndex],SIGNAL(inputDataSignal()),this,triggerSlot));
    }
    else
    {
        return 0;
    }
}
bool Node::connectInternalTrigger(const char * triggerSlot)
{
    if(getInternalTrigger!=NULL)
    {
        QObject * trigger;
        QString triggersignal;
        getInternalTrigger(paramsptr.get(),varsptr.get(),trigger,triggersignal);
        if(bool(connect(trigger,triggersignal.toUtf8().constData(),this,triggerSlot)))
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}

bool Node::disconnectExternalTrigger(QObject * externalTrigger, const char * externalTriggerSignal, const char * triggerSlot)
{
    return bool(disconnect(externalTrigger,externalTriggerSignal,this,triggerSlot));
}

bool Node::disconnectExternalTrigger(int inputPortIndex, const char * triggerSlot)
{
    if(inputPortIndex>=0&&inputPortIndex<inputports.size())
    {
        return bool(disconnect(inputports[inputPortIndex],SIGNAL(inputDataSignal()),this,triggerSlot));
    }
    else
    {
        return 0;
    }
}

bool Node::disconnectInternalTrigger(const char * triggerSlot)
{
    if(getInternalTrigger!=NULL)
    {
        QObject * trigger;
        QString triggersignal;
        getInternalTrigger(paramsptr.get(),varsptr.get(),trigger,triggersignal);
        if(bool(disconnect(trigger,triggersignal.toUtf8().constData(),this,triggerSlot)))
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}

void Node::setInputNodesName(QList<QString> inputNodesName)
{
    inputnodesname=QVector<QString>::fromList(inputNodesName);
    inputnodesname.resize(inputports.size());
}

void Node::setOutputNodesName(QList<QString> outputNodesName)
{
    outputnodesname=QVector<QString>::fromList(outputNodesName);
    outputnodesname.resize(outputports.size());
}

QVector<QString> Node::getInputNodesName()
{
    return inputnodesname;
}

QVector<QString> Node::getOutputNodesName()
{
    return outputnodesname;
}

InputPort * Node::getInputPort(int inputPortIndex)
{
    if(inputPortIndex>=0&&inputPortIndex<inputports.size())
    {
        return inputports[inputPortIndex];
    }
    else
    {
        return NULL;
    }
}

OutputPort * Node::getOutputPort(int outputPortIndex)
{
    if(outputPortIndex>=0&&outputPortIndex<outputports.size())
    {
        return outputports[outputPortIndex];
    }
    else
    {
        return NULL;
    }
}

void Node::nodeTriggerTime(NodeTriggerState nodeTriggerState)
{
    emit nodeTriggerTimeSignal(QDateTime::currentDateTime(),nodeTriggerState);
}

QVector<void *> Node::convertBoostData(QVector<boost::shared_ptr<void> > & boostData)
{
    int i,n=boostData.size();
    QVector<void *> result(n);
    for(i=0;i<n;i++)
    {
        result[i]=boostData[i].get();
    }
    return result;
}

void Node::openNodeSlot()
{
    if(!openflag&&setParamsVarsOpenNode(configname,nodetype,nodeclass,nodename,paramsptr.get(),varsptr.get()))
    {
        openflag=1;
        emit openNodeSignal();
    }
    else if(!openflag)
    {
        emit openNodeErrorSignal();
    }
}

void Node::closeNodeSlot()
{
    if(openflag&&handleVarsCloseNode(paramsptr.get(),varsptr.get()))
    {
        openflag=0;
        int i,n=inputports.size();
        for(i=0;i<n;i++)
        {
            inputports[i]->clear();
        }
        emit closeNodeSignal();
    }
    else if(!openflag)
    {
        emit closeNodeErrorSignal();
    }
}
