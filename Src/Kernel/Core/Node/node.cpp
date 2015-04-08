#include"node.h"

using namespace RobotSDK;

Node::Node(QString nodeClass, QString nodeName, QString libraryFileName, QString configFileName, QString exName)
    : QObject(NULL)
{
    _openflag=0;
    installEventFilter(this);

    _nodeclass=nodeClass;
    _nodename=nodeName;
    _exname=exName;
    _libraryfilename=libraryFileName;
    _configfilename=configFileName;

    uint i,n=_funcptrlist.size();
    _loadflag=1;
    for(i=0;i<n;i++)
    {
        QFunctionPointer funcptr=_funcptrcloadmap[_funcptrlist.at(i)](_libraryfilename,_nodeclass,_exname);
        if(funcptr==NULL)
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
    }

    if(_loadflag)
    {
        _inputportnum=NODE_FUNC_PTR(getInputPortNum);
        _outputportnum=NODE_FUNC_PTR(getOutputPortNum);

        NODE_PARAMS_ARG=NODE_FUNC_PTR(generateNodeParams);
        NODE_PARAMS_ARG->_nodeclass=_nodeclass;
        NODE_PARAMS_ARG->_nodename=_nodename;
        NODE_PARAMS_ARG->_exname=_exname;
        NODE_PARAMS_ARG->loadXMLValues(_configfilename,_nodeclass,_nodename);

        NODE_VARS_ARG=NODE_FUNC_PTR(generateNodeVars);
        NODE_VARS_ARG->_inputportnum=_inputportnum;
        NODE_VARS_ARG->_buffersize.fill(1,_inputportnum);
        NODE_VARS_ARG->_obtaindatabehavior.fill(CopyOldest,_inputportnum);
        NODE_VARS_ARG->_obtaindatasize.fill(0,_inputportnum);
        NODE_VARS_ARG->_triggerflag.fill(0,_inputportnum);
        NODE_VARS_ARG->loadXMLValues(_configfilename,_nodeclass,_nodename);
        NODE_VARS_ARG->nodeSwitcher->_node=this;
        NODE_VARS_ARG->_node=this;

        NODE_DATA_ARG=XML_DATA_BASE_TYPE();
        INPUT_PARAMS_ARG.resize(_inputportnum);
        INPUT_DATA_ARG.resize(_inputportnum);

        _initializeflag=NODE_FUNC_PTR(initializeNode);

        if(_initializeflag)
        {
            _inputports=new InputPorts(_inputportnum, NODE_VARS_ARG);
            connect(&_inputthread, SIGNAL(finished()), _inputports, SLOT(deleteLater()));
            connect(_inputports, SIGNAL(signalObtainParamsData(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE))
                    ,this, SLOT(slotObtainParamsData(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE)), Qt::QueuedConnection);
            _inputports->moveToThread(&_inputthread);
            _inputthread.start();


            _outputports=new OutputPorts(_outputportnum);
            connect(&_outputthread,SIGNAL(finished()),_outputports,SLOT(deleteLater()));
            connect(this, SIGNAL(signalSendParamsData(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE))
                    , _outputports, SLOT(slotSendParamsData(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)), Qt::QueuedConnection);
            _outputports->moveToThread(&_outputthread);
            _outputthread.start();


            QMap< QString, QObject * >::const_iterator triggeriter;
            for(triggeriter=NODE_VARS_ARG->_qobjecttriggermap.begin();triggeriter!=NODE_VARS_ARG->_qobjecttriggermap.end();triggeriter++)
            {
                QList< QString > triggersignals=NODE_VARS_ARG->_defaultconnectionmap.values(triggeriter.value());
                n=triggersignals.size();
                for(i=0;i<n;i++)
                {
                    connect(triggeriter.value(), triggersignals.at(i).toUtf8().data(), this, SLOT(slotDefaultTrigger()));
                }
            }
            for(triggeriter=NODE_VARS_ARG->_qwidgettriggermap.begin();triggeriter!=NODE_VARS_ARG->_qwidgettriggermap.end();triggeriter++)
            {
                QList< QString > triggersignals=NODE_VARS_ARG->_defaultconnectionmap.values(triggeriter.value());
                n=triggersignals.size();
                for(i=0;i<n;i++)
                {
                    connect(triggeriter.value(), triggersignals.at(i).toUtf8().data(), this, SLOT(slotDefaultTrigger()));
                }
            }
            QMultiMap< QObject *, QPair< QString, QString > >::const_iterator userconnection;
            for(userconnection=NODE_VARS_ARG->_userconnectionmap.begin();userconnection!=NODE_VARS_ARG->_userconnectionmap.end();userconnection++)
            {
                connect(userconnection.key(), userconnection.value().first.toUtf8().data(), this, userconnection.value().second.toUtf8().data());
            }
            QMultiMap< QPair< QObject *, QObject * > , QPair< QString, QString > >::const_iterator connection;
            for(connection=NODE_VARS_ARG->_connectionmap.begin();connection!=NODE_VARS_ARG->_connectionmap.end();connection++)
            {
                connect(connection.key().first, connection.value().first.toUtf8().data(), connection.key().second, connection.value().second.toUtf8().data());
            }
            NODE_VARS_ARG->moveTriggerToPoolThread(this, &_poolthread);
            _poolthread.start();
        }
        else
        {
            qDebug()<<QString("[FAILURE] Can not initialize %1::%2.").arg(_nodeclass).arg(_nodename);
        }
    }
    else
    {
        _funcptrmap.clear();
        qDebug()<<QString("[FAILURE] Can not load %1::%2 from %3").arg(_nodeclass).arg(_nodename).arg(_libraryfilename);
    }
}

Node::~Node()
{
    _poolthread.quit();
    _inputthread.quit();
    _outputthread.quit();
    _poolthread.wait();
}

bool Node::eventFilter(QObject *obj, QEvent *ev)
{
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
                _openflag=NODE_FUNC_PTR(openNode);
            }
            else
            {
                _openflag=!NODE_FUNC_PTR(closeNode);
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
            _openflag=NODE_FUNC_PTR(openNode);

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
            _openflag=!NODE_FUNC_PTR(closeNode);

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
    NODE_DATA_ARG=NODE_FUNC_PTR(generateNodeData);
    if(NODE_FUNC_PTR(main))
    {
        emit signalSendParamsData(NODE_PARAMS_ARG,NODE_DATA_ARG);
    }
}

void Node::slotObtainParamsData(PORT_PARAMS_CAPSULE inputParams, PORT_DATA_CAPSULE inputData)
{
    INPUT_PARAMS_ARG=inputParams;
    INPUT_DATA_ARG=inputData;
    NODE_DATA_ARG=NODE_FUNC_PTR(generateNodeData);
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
