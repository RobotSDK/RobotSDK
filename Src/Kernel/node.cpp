#include"node.h"

using namespace RobotSDK;

Node::Node(QString libraryFileName, QString configFileName, QString nodeClass, QString nodeName, QString exName)
    : QObject(NULL)
{
    _openflag=0;
    installEventFilter(this);

    _libraryfilename=libraryFileName;
    _configfilename=configFileName;
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

        NODE_PARAMS_ARG=generateNodeParams();
        NODE_PARAMS_ARG->_nodeclass=_nodeclass;
        NODE_PARAMS_ARG->_nodename=_nodename;
        NODE_PARAMS_ARG->_exname=_exname;
        NODE_PARAMS_ARG->loadXMLValues(_configfilename,_nodeclass,_nodename);

        NODE_VARS_ARG=generateNodeVars();
        NODE_VARS_ARG->_inputportnum=_inputportnum;
        NODE_VARS_ARG->_buffersize.fill(0,_inputportnum);
        NODE_VARS_ARG->_obtaindatabehavior.fill(GrabLatest,_inputportnum);
        NODE_VARS_ARG->_obtaindatasize.fill(1,_inputportnum);
        NODE_VARS_ARG->_triggerflag.fill(1,_inputportnum);
        NODE_VARS_ARG->loadXMLValues(_configfilename,_nodeclass,_nodename);

        NODE_VARS_ARG->nodeSwitcher->_node=this;
        NODE_VARS_ARG->_node=this;
        QPalette pal=NODE_VARS_ARG->nodeSwitcher->palette();
        pal.setColor(QPalette::Button, QColor(_openflag ? Qt::green : Qt::red));
        NODE_VARS_ARG->nodeSwitcher->setAutoFillBackground(1);
        NODE_VARS_ARG->nodeSwitcher->setPalette(pal);
        NODE_VARS_ARG->nodeSwitcher->update();
        NODE_VARS_ARG->nodeSwitcher->setText(QString("%1 %2::%3").arg(_openflag?"Close":"Open").arg(_nodeclass).arg(_nodename));

        NODE_VARS_ARG->widget->setWindowTitle(QString("%1::%2").arg(_nodeclass).arg(_nodename));


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
            _inputports=new InputPorts(_inputportnum, NODE_VARS_ARG);
            _inputthread=std::shared_ptr<QThread>(new QThread);
            connect(_inputthread.get(),SIGNAL(finished()), _inputports, SLOT(deleteLater()));
            connect(_inputports,INPUTPORTS_SIGNAL,this,NODE_SLOT,Qt::QueuedConnection);
            _inputports->moveToThread(_inputthread.get());
            _inputthread->start();


            _outputports=new OutputPorts(_outputportnum);
            _outputthread=std::shared_ptr<QThread>(new QThread);
            connect(_outputthread.get(),SIGNAL(finished()),_outputports,SLOT(deleteLater()));
            connect(this,NODE_SIGNAL, _outputports,OUTPUTPORTS_SLOT,Qt::QueuedConnection);
            _outputports->moveToThread(_outputthread.get());
            _outputthread->start();


            QMap< QString, QObject * >::const_iterator triggeriter;
            for(triggeriter=NODE_VARS_ARG->_qobjecttriggermap.begin();triggeriter!=NODE_VARS_ARG->_qobjecttriggermap.end();triggeriter++)
            {
                QList< QString > triggersignals=NODE_VARS_ARG->_defaultconnectionmap.values(triggeriter.value());
                n=triggersignals.size();
                for(i=0;i<n;i++)
                {
                    connect(triggeriter.value(), triggersignals.at(i).toUtf8().data(), this, SLOT(slotDefaultTrigger()), Qt::QueuedConnection);
                }
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
            _poolthread=std::shared_ptr<QThread>(new QThread);
            NODE_VARS_ARG->moveTriggerToPoolThread(this, _poolthread.get());
            _poolthread->start();
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

Node::~Node()
{    
    _inputthread->quit();
    _poolthread->quit();
    _outputthread->quit();

    _inputthread->wait();
    _poolthread->wait();
    _outputthread->wait();
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
        if(ev->type()==NodeSwitcher::SwitchEventType)
        {
            _openflag=!_openflag;
            if(_openflag)
            {
                NODE_PARAMS_ARG->loadXMLValues(_configfilename,_nodeclass,_nodename);
                NODE_VARS_ARG->loadXMLValues(_configfilename,_nodeclass,_nodename);
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

            QPalette pal=NODE_VARS_ARG->nodeSwitcher->palette();
            pal.setColor(QPalette::Button, QColor(_openflag ? Qt::green : Qt::red));
            NODE_VARS_ARG->nodeSwitcher->setAutoFillBackground(1);
            NODE_VARS_ARG->nodeSwitcher->setPalette(pal);
            NODE_VARS_ARG->nodeSwitcher->update();
            NODE_VARS_ARG->nodeSwitcher->setText(QString("%1 %2::%3").arg(_openflag?"Close":"Open").arg(_nodeclass).arg(_nodename));
            return 1;
        }
        else if(ev->type()==NodeSwitcher::OpenNodeEventType)
        {
            NODE_PARAMS_ARG->loadXMLValues(_configfilename,_nodeclass,_nodename);
            NODE_VARS_ARG->loadXMLValues(_configfilename,_nodeclass,_nodename);
            if(CHECK_NODE_FUNC_PTR(openNode))
            {
                _openflag=NODE_FUNC_PTR(openNode);
            }
            else
            {
                _openflag=1;
            }

            QPalette pal=NODE_VARS_ARG->nodeSwitcher->palette();
            pal.setColor(QPalette::Button, QColor(_openflag ? Qt::green : Qt::red));
            NODE_VARS_ARG->nodeSwitcher->setAutoFillBackground(1);
            NODE_VARS_ARG->nodeSwitcher->setPalette(pal);
            NODE_VARS_ARG->nodeSwitcher->update();
            NODE_VARS_ARG->nodeSwitcher->setText(QString("%1 %2::%3").arg(_openflag?"Close":"Open").arg(_nodeclass).arg(_nodename));
            return 1;
        }
        else if(ev->type()==NodeSwitcher::CloseNodeEventType)
        {
            if(CHECK_NODE_FUNC_PTR(closeNode))
            {
                _openflag=!NODE_FUNC_PTR(closeNode);
            }
            else
            {
                _openflag=0;
            }
            QPalette pal=NODE_VARS_ARG->nodeSwitcher->palette();
            pal.setColor(QPalette::Button, QColor(_openflag ? Qt::green : Qt::red));
            NODE_VARS_ARG->nodeSwitcher->setAutoFillBackground(1);
            NODE_VARS_ARG->nodeSwitcher->setPalette(pal);
            NODE_VARS_ARG->nodeSwitcher->update();
            NODE_VARS_ARG->nodeSwitcher->setText(QString("%1 %2::%3").arg(_openflag?"Close":"Open").arg(_nodeclass).arg(_nodename));
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

InputPort * Node::getInputPort(uint portID)
{
    if(portID<_inputports->portnum)
    {
        return _inputports->inputports[portID];
    }
    else
    {
        return NULL;
    }
}

OutputPort * Node::getOutputPort(uint portID)
{
    if(portID<_outputports->portnum)
    {
        return _outputports->outputports[portID];
    }
    else
    {
        return NULL;
    }
}
