#ifndef PORT_H
#define PORT_H

#include<Core/ModuleDev/defines.h>
#include<Core/ModuleDev/valuebase.h>

namespace RobotSDK
{

class InputPort : public QObject
{
    Q_OBJECT
public:
    InputPort(uint portID, QObject * parent);
protected:
    uint portid;
public slots:
    void slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE inputParams, TRANSFER_PORT_DATA_TYPE inputData);
signals:
    void signalReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE inputParams, TRANSFER_PORT_DATA_TYPE inputData, uint inputPortID);
};

class OutputPort : public QObject
{
    Q_OBJECT
public:
    OutputPort(uint portID, QObject * parent);
protected:
    uint portid;
public slots:
    void slotSendParamsData(QList< bool > filterFlag, TRANSFER_NODE_PARAMS_TYPE outputParams, TRANSFER_NODE_DATA_TYPE outputData);
signals:
    void signalSendParamsData(TRANSFER_PORT_PARAMS_TYPE outputParams, TRANSFER_PORT_DATA_TYPE outputData);
};

//========================================================================

class InputPorts : public QObject
{
    Q_OBJECT
public:
    InputPorts(uint portNum, TRANSFER_NODE_VARS_TYPE nodeVars);
protected:
    uint portnum;
    QVector< PORT_PARAMS_BUFFER > portparamsbuffer;
    QVector< PORT_DATA_BUFFER > portdatabuffer;
    QVector< uint > buffercount;
    TRANSFER_NODE_VARS_TYPE nodevars;
public slots:
    void slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE inputParams, TRANSFER_PORT_DATA_TYPE inputData, uint inputPortID);
    void slotClearBuffer();
signals:
    void signalObtainParamsData(PORT_PARAMS_CAPSULE inputParams, PORT_DATA_CAPSULE inputData);
};

class OutputPorts : QObject
{
    Q_OBJECT
public:
    OutputPorts(uint portNum);
    ~OutputPorts();
public slots:
    void slotSendParamsData(QList< bool > filterFlag, TRANSFER_NODE_PARAMS_TYPE outputParams, TRANSFER_NODE_DATA_TYPE outputData);
signals:
    void signalSendParamsData(QList< bool > filterFlag, TRANSFER_NODE_PARAMS_TYPE outputParams, TRANSFER_NODE_DATA_TYPE outputData);
};

}

#endif // PORT_H

