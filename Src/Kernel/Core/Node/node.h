#ifndef NODE_H
#define NODE_H

#include<Core/ModuleDev/defines.h>
#include<Core/ModuleDev/valuebase.h>
#include<Core/Port/port.h>

namespace RobotSDK
{

class Node : public QObject
{
    Q_OBJECT
public:
    Node(QString nodeClass, QString nodeName, QString libraryFileName, QString configFileName, QString exName=QString());
    ~Node();
private:
    QString _nodeclass;
    QString _nodename;
    QString _exname;
private:
    QThread _inputthread;
    QThread _poolthread;
    QThread _outputthread;
private:
    InputPorts * _inputports;
    OutputPorts * _outputports;
protected:
    QString _libraryfilename;
    QList< QString > _funcptrlist;
    QMap< QString, std::function< QFunctionPointer(QString, QString, QString) > > _funcptrcloadmap;
    QMap< QString, QFunctionPointer > _funcptrmap;
protected:
    uint _inputportnum;
    uint _outputportnum;
protected:
    bool _loadflag;
    bool _initializeflag;
    bool _openflag;
protected:
    QString _configfilename;
    XML_PARAMS_BASE_TYPE NODE_PARAMS_ARG;
    XML_VARS_BASE_TYPE NODE_VARS_ARG;
protected:
    XML_DATA_BASE_TYPE NODE_DATA_ARG;
protected:
    PORT_PARAMS_CAPSULE INPUT_PARAMS_ARG;
    PORT_DATA_CAPSULE INPUT_DATA_ARG;
protected:
    bool eventFilter(QObject * obj, QEvent * ev);
private slots:
    void slotDefaultTrigger();
    void slotObtainParamsData(PORT_PARAMS_CAPSULE inputParams, PORT_DATA_CAPSULE inputData);
signals:
    void signalSendParamsData(TRANSFER_NODE_PARAMS_TYPE outputParams, TRANSFER_NODE_DATA_TYPE outputData);
protected:
    InputPort * getInputPort(uint portID);
    OutputPort * getOutputPort(uint portID);
public:
    ADD_NODE_DEFAULT_FUNC_PTR(uint, getInputPortNum)
    ADD_NODE_DEFAULT_FUNC_PTR(uint, getOutputPortNum)
    ADD_NODE_DEFAULT_FUNC_PTR(XML_PARAMS_BASE_TYPE, generateNodeParams)
    ADD_NODE_DEFAULT_FUNC_PTR(XML_VARS_BASE_TYPE, generateNodeVars)
    ADD_NODE_DEFAULT_FUNC_PTR(XML_DATA_BASE_TYPE, generateNodeData)
    ADD_NODE_FUNC_PTR(bool, initializeNode)
    ADD_NODE_FUNC_PTR(bool, openNode)
    ADD_NODE_FUNC_PTR(bool, closeNode)
    ADD_NODE_FUNC_PTR(bool, main)
};

}

#endif // NODE_H

