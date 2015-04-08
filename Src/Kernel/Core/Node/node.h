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
    Node(QString nodeClass, QString nodeName);
    ~Node();
private:
    QString _nodeclass;
    QString _nodename;
private:
    QThread _inputthread;
    InputPorts * _inputports;
    QThread _outputthread;
    OutputPorts * _outputports;
protected:
    XML_PARAMS_BASE_TYPE NODE_PARAMS_ARG;
    XML_VARS_BASE_TYPE NODE_VARS_ARG;
    XML_DATA_BASE_TYPE NODE_DATA_ARG;
    PORT_PARAMS_CAPSULE INPUT_PARAMS_ARG;
    PORT_DATA_CAPSULE INPUT_DATA_ARG;
protected:
    QString _libraryfilename;
    QMap< QString, std::function< QFunctionPointer(QLibrary &, QString, QString) > > _funcloadmap;
    QMap< QString, QFunctionPointer > _funcmap;
private slots:
    void slotInitialNode(QString );
    void slotObtainParamsData(PORT_PARAMS_CAPSULE inputParams, PORT_DATA_CAPSULE inputData);
    void slotDefaultTrigger();
public:
    ADD_NODE_DEFAULT_FUNC_PTR(uint, getInputPortNum)
    ADD_NODE_DEFAULT_FUNC_PTR(uint, getOutputPortNum)
    ADD_NODE_DEFAULT_FUNC_PTR(XML_PARAMS_BASE_TYPE, generateNodeParams)
    ADD_NODE_DEFAULT_FUNC_PTR(XML_VARS_BASE_TYPE, generateNodeVars)
    ADD_NODE_DEFAULT_FUNC_PTR(XML_DATA_BASE_TYPE, generateNodeData)
    ADD_NODE_FUNC_PTR(bool, initialNode)
    ADD_NODE_FUNC_PTR(bool, openNode)
    ADD_NODE_FUNC_PTR(bool, closeNode)
    ADD_NODE_FUNC_PTR(bool, main)
};

}

#endif // NODE_H

