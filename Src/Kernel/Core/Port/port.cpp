#include"port.h"

using namespace RobotSDK;

InputPort::InputPort(uint portID, QObject *parent)
    : QObject(parent)
{
    portid=portID;
}

void InputPort::slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE inputParams, TRANSFER_PORT_DATA_TYPE inputData)
{
    emit signalReceiveParamsData(inputParams, inputData, portid);
}

OutputPort::OutputPort(uint portID, QObject *parent)
    : QObject(parent)
{
    portid=portID;
}

void OutputPort::slotSendParamsData(TRANSFER_NODE_PARAMS_TYPE outputParams, TRANSFER_NODE_DATA_TYPE outputData)
{
    if(portid>=uint(outputData->_filterflag.size())||outputData->_filterflag.at(portid))
    {
        outputData->portid=portid;
        emit signalSendParamsData(outputParams, outputData);
    }
}

InputPorts::InputPorts(uint portNum, TRANSFER_NODE_VARS_TYPE nodeVars)
    : QObject(NULL)
{
    portnum=portNum;
    inputports.resize(portnum);
    portparamsbuffer.resize(portnum);
    portdatabuffer.resize(portnum);
    buffercount.resize(portnum);
    buffercount.fill(0);
    uint i;
    for(i=0;i<portnum;i++)
    {
        InputPort * port=new InputPort(i,this);
        connect(port, SIGNAL(signalReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE,TRANSFER_PORT_DATA_TYPE,uint)),
                this, SLOT(slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE,TRANSFER_PORT_DATA_TYPE,uint)),Qt::DirectConnection);
        inputports[i]=port;
    }
    nodevars=nodeVars;
}

void InputPorts::slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE inputParams, TRANSFER_PORT_DATA_TYPE inputData, uint inputPortID)
{
    portparamsbuffer[inputPortID].push_front(inputParams);
    portdatabuffer[inputPortID].push_front(inputData);
    buffercount[inputPortID]++;

    QMutexLocker locker(&(nodevars->_inputportlock));
    QVector< uint > _buffersize=nodevars->_buffersize;
    QVector< ObtainBehavior > _obtaindatabehavior=nodevars->_obtaindatabehavior;
    QVector< uint > _obtaindatasize=nodevars->_obtaindatasize;
    QVector< bool > _triggerflag=nodevars->_triggerflag;
    locker.unlock();

    if(_buffersize.at(inputPortID)>0&&buffercount[inputPortID]>_buffersize.at(inputPortID))
    {
        portparamsbuffer[inputPortID].pop_back();
        portdatabuffer[inputPortID].pop_back();
        buffercount[inputPortID]--;
    }

    if(!_triggerflag.at(inputPortID))
    {
        return;
    }

    uint i;
    bool sendFlag=1;
    for(i=0;i<portnum&&sendFlag;i++)
    {
        if((0b100)&(_obtaindatabehavior.at(i)))
        {
            sendFlag=_obtaindatasize.at(i)==0||buffercount[i]>=_obtaindatasize.at(i);
        }
    }
    if(sendFlag)
    {
        PORT_PARAMS_CAPSULE inputparams;
        inputparams.resize(portnum);
        PORT_DATA_CAPSULE inputdata;
        inputdata.resize(portnum);
        for(i=0;i<portnum;i++)
        {
            ObtainBehavior obtaindatabehavior=_obtaindatabehavior.at(i);
            uint obtaindatasize=_obtaindatasize.at(i);
            if((0b010)&(obtaindatabehavior))
            {
                if(obtaindatasize>0)
                {
                    std::copy(portparamsbuffer[i].begin(), portparamsbuffer[i].begin()+obtaindatasize, inputparams[i].begin());
                    std::copy(portdatabuffer[i].begin(), portdatabuffer[i].begin()+obtaindatasize, inputdata[i].begin());
                }
                else
                {
                    inputparams[i]=portparamsbuffer[i];
                    inputdata[i]=portdatabuffer[i];
                }
                if((0b001)&(_obtaindatabehavior.at(i)))
                {
                    portparamsbuffer[i].clear();
                    portdatabuffer[i].clear();
                    buffercount[i]=0;
                }
            }
            else
            {
                if(obtaindatasize>0)
                {
                    std::copy(portparamsbuffer[i].end()-obtaindatasize, portparamsbuffer[i].end(), inputparams[i].begin());
                    std::copy(portdatabuffer[i].end()-obtaindatasize, portdatabuffer[i].end(), inputdata[i].begin());
                }
                else
                {
                    inputparams[i]=portparamsbuffer[i];
                    inputdata[i]=portdatabuffer[i];
                }
                if((0b001)&(_obtaindatabehavior.at(i)))
                {
                    if(obtaindatasize>0)
                    {
                        while(obtaindatasize-->0)
                        {
                            portparamsbuffer[i].pop_back();
                            portdatabuffer[i].pop_back();
                            buffercount[i]--;
                        }
                    }
                    else
                    {
                        portparamsbuffer[i].clear();
                        portdatabuffer[i].clear();
                        buffercount[i]=0;
                    }
                }
            }
        }
        emit signalObtainParamsData(inputparams,inputdata);
    }
}

void InputPorts::slotClearBuffer()
{
    uint i;
    for(i=0;i<portnum;i++)
    {
        portparamsbuffer[i].clear();
        portdatabuffer[i].clear();
        buffercount[i]=0;
    }
}

OutputPorts::OutputPorts(uint portNum)
    : QObject(NULL)
{
    portnum=portNum;
    outputports.resize(portnum);
    uint i;
    for(i=0;i<portnum;i++)
    {
        OutputPort * port=new OutputPort(i,this);
        connect(this, SIGNAL(signalSendParamsData(TRANSFER_NODE_PARAMS_TYPE,TRANSFER_NODE_DATA_TYPE)),
                port, SLOT(slotSendParamsData(TRANSFER_NODE_PARAMS_TYPE,TRANSFER_NODE_DATA_TYPE)),Qt::DirectConnection);
        outputports[i]=port;
    }
}

void OutputPorts::slotSendParamsData(TRANSFER_NODE_PARAMS_TYPE outputParams, TRANSFER_NODE_DATA_TYPE outputData)
{
    emit signalSendParamsData(outputParams,outputData);
}
