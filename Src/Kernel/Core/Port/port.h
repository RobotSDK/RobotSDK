#ifndef PORT_H
#define PORT_H

#include<QObject>
#include<QList>
#include<QMutex>
#include<QMutexLocker>
#include<memory>
#include<Core/ModuleDev/valuebase.h>
#include<Core/ModuleDev/defines.h>

namespace RobotSDK
{

class InputPort : public QObject
{
    Q_OBJECT
public:
    InputPort(int portID, QObject * parent);
protected:
    unsigned int portid;
signals:
    void signalReceiveParamsData(unsigned int outputPortID, TRANSFER_CONST_TYPE inputParams, TRANSFER_CONST_TYPE inputData, unsigned int inputPortID);
public slots:
    void slotReceiveParamsData(unsigned int outputPortID, TRANSFER_CONST_TYPE inputParams, TRANSFER_CONST_TYPE inputData);
};

class OutputPort : public QObject
{
    Q_OBJECT
public:
    OutputPort(int portID, QObject * parent);
protected:
    unsigned int portid;
signals:
    void signalSendParamsData(unsigned int outputPortID, TRANSFER_CONST_TYPE outputParams, TRANSFER_CONST_TYPE outputData);
public slots:
    void slotSendParamsData(TRANSFER_CONST_TYPE outputParams, TRANSFER_CONST_TYPE outputData);
};

//========================================================================

class InputPorts : public QObject
{
public:
    InputPorts(unsigned int portNum, std::shared_ptr<XMLVarsBase> nodeVars);
    ~InputPorts();
protected:
    QList< _PORT_BUFFER > portparamsbuffer;
    QList< _PORT_BUFFER > portdatabuffer;
    QThread thread;
    std::shared_ptr<XMLVarsBase> nodevars;
signals:
    void signalObtainParamsData(OBTAIN_CAPSULE inputParams, OBTAIN_CAPSULE inputData);
public slots:
    void slotReceiveParamsData(unsigned int outputPortID, TRANSFER_CONST_TYPE inputParams, TRANSFER_CONST_TYPE inputData, unsigned int inputPortID);
};

class OutputPorts : QObject
{
public:
    OutputPorts(unsigned int portNum);
    ~OutputPorts();
public:
    QThread thread;
};

//========================================================================


class Ports
{
protected:

protected:
    QThread outputthread;
    QList< OutputPort * > outputports;
protected:
    void setInputPortNum(unsigned int portNum);
    void setOutputPortNum(unsigned int portNum);
protected:
    bool obtainInputData(OBTAIN_CAPSULE & inputParams, OBTAIN_CAPSULE & outputParams);
};

#endif // PORT_H

