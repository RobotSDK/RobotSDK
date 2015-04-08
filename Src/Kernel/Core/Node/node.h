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
    Node(QString libraryFileName, QString nodeClass, QString nodeName, QString configFileName);
    ~Node();
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
    void slotObtainParamsData(PORT_PARAMS_CAPSULE inputParams, PORT_DATA_CAPSULE inputData);
    void slotDefaultTrigger();
public:
    void loadFunctions();
public:
    ADD_NODE_FUNC_PTR(uint,getInputPortNum)

};

}

#endif // NODE_H

