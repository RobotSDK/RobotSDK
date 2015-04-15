#ifndef NODE_H
#define NODE_H

#include<port.h>

namespace RobotSDK
{

class Node : public QObject
{
    Q_OBJECT
public:
    Node(QString libraryFileName, QString configFileName, QString nodeFullName);
    ~Node();
public:
    QString _libraryfilename;
    QString _configfilename;
    QString _nodefullname;
    QString _nodeclass;
    QString _nodename;
    QString _exname;
    uint _inputportnum;
    uint _outputportnum;
    bool _loadflag;
    bool _initializeflag;
    bool _openflag;
    XML_PARAMS_BASE_TYPE NODE_PARAMS_ARG;
    XML_VARS_BASE_TYPE NODE_VARS_ARG;
private:
    std::shared_ptr<QThread> _inputthread;
    std::shared_ptr<QThread> _poolthread;
    std::shared_ptr<QThread> _outputthread;
private:
    InputPorts * _inputports;
    OutputPorts * _outputports;
protected:
    QList< QString > _funcptrlist;
    QList< bool > _funcptrmandatoryflaglist;
    QMap< QString, std::function< QFunctionPointer(QString, QString, QString) > > _funcptrcloadmap;
    QMap< QString, QFunctionPointer > _funcptrmap;
    QMap< QString,  bool > _funcptrflag;
protected:
    PORT_PARAMS_CAPSULE INPUT_PARAMS_ARG;
    PORT_DATA_CAPSULE INPUT_DATA_ARG;
    XML_DATA_BASE_TYPE NODE_DATA_ARG;
protected:
    bool eventFilter(QObject * obj, QEvent * ev);
protected slots:
    void slotDefaultTrigger();
    void slotObtainParamsData(PORT_PARAMS_CAPSULE inputParams, PORT_DATA_CAPSULE inputData);
signals:
    void signalSendParamsData(TRANSFER_NODE_PARAMS_TYPE outputParams, TRANSFER_NODE_DATA_TYPE outputData);
    void signalNodeState(bool openFlag, QString nodeFullName);
public:
    const InputPort * getInputPort(uint portID);
    const OutputPort * getOutputPort(uint portID);
protected:
    typedef uint (*getInputPortNumFptr)();
    getInputPortNumFptr getInputPortNum;
    typedef uint (*getOutputPortNumFptr)();
    getOutputPortNumFptr getOutputPortNum;
    typedef XML_PARAMS_BASE_TYPE (*generateNodeParamsFptr)();
    generateNodeParamsFptr generateNodeParams;
    typedef XML_VARS_BASE_TYPE (*generateNodeVarsFptr)();
    generateNodeVarsFptr generateNodeVars;
    typedef XML_DATA_BASE_TYPE (*generateNodeDataFptr)();
    generateNodeDataFptr generateNodeData;
    ADD_NODE_FUNC_PTR(bool, initializeNode, 0)
    ADD_NODE_FUNC_PTR(bool, openNode, 0)
    ADD_NODE_FUNC_PTR(bool, closeNode, 0)
    ADD_NODE_FUNC_PTR(bool, main, 1)
};

#define USE_DEFAULT_NODE \
    extern "C" RobotSDK_EXPORT void * NODE_FUNC_NAME(generateNode) \
    (QString libraryFileName, QString configFileName, QString nodeFullName){ \
    return static_cast<void *>(new Node(libraryFileName, configFileName, nodeFullName));} NODE_DEFAULT_FUNC

#define USE_EXTENDED_NODE(nodeType, ...) \
    extern "C" RobotSDK_EXPORT void * NODE_FUNC_NAME(generateNode) \
    (QString libraryFileName, QString configFileName, QString nodeFullName){ \
    return static_cast<void *>(new nodeType(libraryFileName, configFileName, nodeFullName, ##__VA_ARGS__));} NODE_DEFAULT_FUNC

}

#endif // NODE_H

