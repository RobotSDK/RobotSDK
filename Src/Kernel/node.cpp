#include"node.h"

using namespace RobotSDK;

Node::Node(QString libraryFileName, QString configFileName, QString nodeFullName)
    : QObject(NULL)
{
    _openflag=0;
    installEventFilter(this);

    _libraryfilename=libraryFileName;
    _configfilename=configFileName;

    QStringList nodenamelist=nodeFullName.split(QString("::"),QString::SkipEmptyParts);
    if(nodenamelist.size()<2||nodenamelist.size()>3)
    {
        qDebug()<<QString("%1 is not a valid node full name (NodeClass::NodeName[::ExName]).");
        _loadflag=0;
        _initializeflag=0;
    }
    else
    {
        QString nodeClass=nodenamelist.at(0);
        QString nodeName=nodenamelist.at(1);
        QString exName=QString();
        if(nodenamelist.size()==3)
        {
            exName=nodenamelist.at(2);
        }

        _nodefullname=nodeFullName;
        _nodeclass=nodeClass;
        _nodename=nodeName;
        _exname=exName;

        uint i,n=_funcptrlist.size();
        _loadflag=1;

        getInputPortNum=(getInputPortNumFptr)LOAD_NODE_FUNC_PTR(_libraryfilename, _nodeclass, getInputPortNum);
        getOutputPortNum=(getOutputPortNumFptr)LOAD_NODE_FUNC_PTR(_libraryfilename, _nodeclass, getOutputPortNum);
        generateNodeParams=(generateNodeParamsFptr)LOAD_NODE_FUNC_PTR(_libraryfilename, _nodeclass, generateNodeParams);
        generateNodeVars=(generateNodeVarsFptr)LOAD_NODE_FUNC_PTR(_libraryfilename, _nodeclass, generateNodeVars);
        generateNodeData=(generateNodeDataFptr)LOAD_NODE_FUNC_PTR(_libraryfilename, _nodeclass, generateNodeData);

        if(getInputPortNum==NULL||getOutputPortNum==NULL||generateNodeParams==NULL||generateNodeVars==NULL||generateNodeData==NULL)
        {
            qDebug()<<QString("Can not resolve default functions from %1. May lack of USE_DEFAULT_NODE or USE_EXTENDED_NODE in module source code.").arg(_libraryfilename);
            _loadflag=0;
        }

        for(i=0;i<n;i++)
        {
            QFunctionPointer funcptr=_funcptrcloadmap[_funcptrlist.at(i)](_libraryfilename,_nodeclass,_exname);
            if(funcptr==NULL&&_funcptrmandatoryflaglist.at(i))
            {
                if(_exname.size()==0)
                {
                    _loadflag=0;
                    qDebug()<<QString("[ERROR] Can not resolve %1__%2 in %3").arg(_nodeclass).arg(_funcptrlist.at(i)).arg(_libraryfilename);
                }
                else
                {
                    _loadflag=0;
                    qDebug()<<QString("[ERROR] Can not resolve %1__%2__%3 and %1__%2 in %4").arg(_nodeclass).arg(_funcptrlist.at(i)).arg(_exname).arg(_libraryfilename);
                }
            }
            _funcptrmap.insert(_funcptrlist.at(i), funcptr);
            _funcptrflag.insert(_funcptrlist.at(i), funcptr!=NULL);
        }

        if(_loadflag)
        {
            _inputportnum=getInputPortNum();
            _outputportnum=getOutputPortNum();

            _inputnodeclass.resize(_inputportnum);
            for(i=0;i<_inputportnum;i++)
            {
                QString inputnodesymbol=QString("%1_INPUT_NODE_%2_ClassName").arg(_nodeclass).arg(i);
                QString * inputnodeclass=(QString *)(QLibrary::resolve(_libraryfilename,inputnodesymbol.toUtf8().data()));
                _inputnodeclass[i]=*inputnodeclass;
            }

            NODE_PARAMS_ARG=generateNodeParams();
            NODE_PARAMS_ARG->nodeclass=_nodeclass;
            NODE_PARAMS_ARG->nodename=_nodename;
            NODE_PARAMS_ARG->exname=_exname;
            NODE_PARAMS_ARG->loadXMLValues(_configfilename,_nodefullname);

            NODE_VARS_ARG=generateNodeVars();
            NODE_VARS_ARG->_inputportnum=_inputportnum;
            NODE_VARS_ARG->_buffersize.fill(0,_inputportnum);
            NODE_VARS_ARG->_obtaindatabehavior.fill(GrabLatest,_inputportnum);
            NODE_VARS_ARG->_obtaindatasize.fill(1,_inputportnum);
            NODE_VARS_ARG->_triggerflag.fill(1,_inputportnum);
            NODE_VARS_ARG->_guithreadflag=0;
            NODE_VARS_ARG->_showwidgetflag=0;
            NODE_VARS_ARG->loadXMLValues(_configfilename,_nodefullname);

            NODE_VARS_ARG->nodeSwitcher->_node=this;
            NODE_VARS_ARG->_node=this;
            connect(this,SIGNAL(signalNodeState(bool,QString)),NODE_VARS_ARG->nodeSwitcher,SLOT(slotNodeState(bool,QString)));

            emit signalNodeState(_openflag,_nodefullname);

            NODE_VARS_ARG->widget->setWindowTitle(_nodefullname);


            NODE_DATA_ARG=XML_DATA_BASE_TYPE();

            if(CHECK_NODE_FUNC_PTR(initializeNode))
            {
                _initializeflag=NODE_FUNC_PTR(initializeNode);
            }
            else
            {
                _initializeflag=1;
            }

            if(_initializeflag)
            {
                if(_inputportnum>0)
                {
                    _inputports=new InputPorts(_inputportnum, NODE_VARS_ARG);
                    _inputthread=std::shared_ptr<QThread>(new QThread);
                    connect(_inputthread.get(),SIGNAL(finished()), _inputports, SLOT(deleteLater()));
                    connect(_inputports,INPUTPORTS_SIGNAL,this,NODE_SLOT,Qt::QueuedConnection);
                    _inputports->moveToThread(_inputthread.get());
                    _inputthread->start();
                }
                else
                {
                    _inputports=NULL;
                    _inputthread=std::shared_ptr<QThread>();
                }

                if(_outputportnum>0)
                {
                    _outputports=new OutputPorts(_outputportnum);
                    _outputthread=std::shared_ptr<QThread>(new QThread);
                    connect(_outputthread.get(),SIGNAL(finished()),_outputports,SLOT(deleteLater()));
                    connect(this,NODE_SIGNAL, _outputports,OUTPUTPORTS_SLOT,Qt::QueuedConnection);
                    _outputports->moveToThread(_outputthread.get());
                    _outputthread->start();
                }
                else
                {
                    _outputports=NULL;
                    _outputthread=std::shared_ptr<QThread>();
                }

                _poolthread=std::shared_ptr<QThread>(new QThread);
                QMap< QString, QObject * >::const_iterator triggeriter;
                bool openpoolflag=0;
                for(triggeriter=NODE_VARS_ARG->_qobjecttriggermap.begin();triggeriter!=NODE_VARS_ARG->_qobjecttriggermap.end();triggeriter++)
                {
                    if(triggeriter.value()->thread()==this->thread())
                    {
                        if(NODE_VARS_ARG->_qobjecttriggerpoolthreadflagmap[triggeriter.key()])
                        {
                            openpoolflag=1;
                            triggeriter.value()->moveToThread(_poolthread.get());
                            connect(_poolthread.get(),SIGNAL(finished()),triggeriter.value(),SLOT(deleteLater()));
                        }
                        else
                        {
                            triggeriter.value()->setParent(this);
                        }
                    }
                    else
                    {
                        NODE_VARS_ARG->_qobjecttriggerpoolthreadflagmap[triggeriter.key()]=0;
                    }
                    QList< QString > triggersignals=NODE_VARS_ARG->_defaultconnectionmap.values(triggeriter.value());
                    n=triggersignals.size();
                    for(i=0;i<n;i++)
                    {
                        if(NODE_VARS_ARG->_qobjecttriggerpoolthreadflagmap[triggeriter.key()])
                        {
                            connect(triggeriter.value(), triggersignals.at(i).toUtf8().data(), this, SLOT(slotDefaultTrigger()), Qt::QueuedConnection);
                        }
                        else
                        {
                            connect(triggeriter.value(), triggersignals.at(i).toUtf8().data(), this, SLOT(slotDefaultTrigger()), Qt::DirectConnection);
                        }
                    }
                }
                if(openpoolflag)
                {
                    _poolthread->start();
                }
                else
                {
                    _poolthread=std::shared_ptr<QThread>();
                }

                for(triggeriter=NODE_VARS_ARG->_qwidgettriggermap.begin();triggeriter!=NODE_VARS_ARG->_qwidgettriggermap.end();triggeriter++)
                {
                    QList< QString > triggersignals=NODE_VARS_ARG->_defaultconnectionmap.values(triggeriter.value());
                    n=triggersignals.size();
                    for(i=0;i<n;i++)
                    {
                        connect(triggeriter.value(), triggersignals.at(i).toUtf8().data(), this, SLOT(slotDefaultTrigger()), Qt::QueuedConnection);
                    }
                }
                QMultiMap< QObject *, QPair< QString, QString > >::const_iterator userconnection;
                for(userconnection=NODE_VARS_ARG->_userconnectionmap.begin();userconnection!=NODE_VARS_ARG->_userconnectionmap.end();userconnection++)
                {
                    connect(userconnection.key(), userconnection.value().first.toUtf8().data(), this, userconnection.value().second.toUtf8().data(), Qt::QueuedConnection);
                }
                QMultiMap< QPair< QObject *, QObject * > , QPair< QString, QString > >::const_iterator connection;
                for(connection=NODE_VARS_ARG->_connectionmap.begin();connection!=NODE_VARS_ARG->_connectionmap.end();connection++)
                {
                    connect(connection.key().first, connection.value().first.toUtf8().data(), connection.key().second, connection.value().second.toUtf8().data());
                }

            }
            else
            {
                if(_exname.size()==0)
                {
                    qDebug()<<QString("[FAILURE] Can not initialize %1::%2.").arg(_nodeclass).arg(_nodename);
                }
                else
                {
                    qDebug()<<QString("[FAILURE] Can not initialize %1::%2::%3.").arg(_nodeclass).arg(_nodename).arg(_exname);
                }
            }
        }
        else
        {
            _funcptrmap.clear();
            if(_exname.size()==0)
            {
                qDebug()<<QString("[FAILURE] Can not load %1::%2 from %3").arg(_nodeclass).arg(_nodename).arg(_libraryfilename);
            }
            else
            {
                qDebug()<<QString("[FAILURE] Can not load %1::%2::%3 from %4").arg(_nodeclass).arg(_nodename).arg(_exname).arg(_libraryfilename);
            }
        }
    }
}

Node::~Node()
{    
    if(_inputthread!=NULL)
    {
        _inputthread->quit();
    }
    if(_poolthread!=NULL)
    {
        _poolthread->quit();
    }
    if(_outputthread!=NULL)
    {
        _outputthread->quit();
    }


    if(_inputthread!=NULL)
    {
        _inputthread->wait();
    }
    if(_poolthread!=NULL)
    {
        _poolthread->wait();
    }
    if(_outputthread!=NULL)
    {
        _outputthread->wait();
    }
}

bool Node::eventFilter(QObject *obj, QEvent *ev)
{
    Q_UNUSED(obj);
    if(!_loadflag||!_initializeflag)
    {
        return 1;
    }
    if(!_openflag&&ev->type()==QEvent::MetaCall)
    {
        return 1;
    }
    else
    {
        int type=int(ev->type());
        if(type==SwitchEventType)
        {
            _openflag=!_openflag;
            if(_openflag)
            {
                NODE_PARAMS_ARG->loadXMLValues(_configfilename,_nodefullname);
                NODE_VARS_ARG->loadXMLValues(_configfilename,_nodefullname);
                if(CHECK_NODE_FUNC_PTR(openNode))
                {
                    _openflag=NODE_FUNC_PTR(openNode);
                }
            }
            else
            {
                if(CHECK_NODE_FUNC_PTR(closeNode))
                {
                    _openflag=!NODE_FUNC_PTR(closeNode);
                }
            }
            emit signalNodeState(_openflag,_nodefullname);
            return 1;
        }
        else if(type==OpenNodeEventType&&_openflag==0)
        {
            NODE_PARAMS_ARG->loadXMLValues(_configfilename,_nodefullname);
            NODE_VARS_ARG->loadXMLValues(_configfilename,_nodefullname);
            if(CHECK_NODE_FUNC_PTR(openNode))
            {
                _openflag=NODE_FUNC_PTR(openNode);
            }
            else
            {
                _openflag=1;
            }
            emit signalNodeState(_openflag,_nodefullname);
            return 1;
        }
        else if(type==CloseNodeEventType&&_openflag==1)
        {
            if(CHECK_NODE_FUNC_PTR(closeNode))
            {
                _openflag=!NODE_FUNC_PTR(closeNode);
            }
            else
            {
                _openflag=0;
            }
            emit signalNodeState(_openflag,_nodefullname);
            return 1;
        }
        else
        {
            return 0;
        }
    }
}

void Node::slotDefaultTrigger()
{
    INPUT_PARAMS_ARG.clear();
    INPUT_DATA_ARG.clear();
    NODE_DATA_ARG=generateNodeData();
    if(NODE_FUNC_PTR(main))
    {
        emit signalSendParamsData(NODE_PARAMS_ARG,NODE_DATA_ARG);
    }
}

void Node::slotObtainParamsData(PORT_PARAMS_CAPSULE inputParams, PORT_DATA_CAPSULE inputData)
{
    INPUT_PARAMS_ARG=inputParams;
    INPUT_DATA_ARG=inputData;
    NODE_DATA_ARG=generateNodeData();
    if(NODE_FUNC_PTR(main))
    {
        emit signalSendParamsData(NODE_PARAMS_ARG,NODE_DATA_ARG);
    }
}

const InputPort *Node::getInputPort(uint portID)
{
    if(_inputports!=NULL&&portID<_inputports->portnum)
    {
        return _inputports->inputports[portID];
    }
    else
    {
        return NULL;
    }
}

const OutputPort *Node::getOutputPort(uint portID)
{
    if(_outputports!=NULL&&portID<_outputports->portnum)
    {
        return _outputports->outputports[portID];
    }
    else
    {
        return NULL;
    }
}
