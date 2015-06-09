#include"sync.h"

using namespace RobotSDK;

Sync::Sync(uint portNum, uint basePortID)
{
    paramsbuffer.resize(portNum);
    databuffer.resize(portNum);
    syncparams.resize(portNum);
    syncdata.resize(portNum);
    deltatime.resize(portNum);
    baseportid=basePortID;
    if(baseportid>=portNum)
    {
        baseportid=0;
    }
    syncrecordid=0;
}

bool Sync::addParamsData(PORT_PARAMS_CAPSULE & inputParams, PORT_DATA_CAPSULE & inputData)
{
    uint i,n=inputParams.size();
    for(i=0;i<n;i++)
    {
        if(inputData[i].size()>0)
        {
            if(databuffer[i].size()>0)
            {
                if(databuffer[i].front()->timestamp<=inputData[i].back()->timestamp)
                {
                    paramsbuffer[i]=inputParams[i]+paramsbuffer[i];
                    databuffer[i]=inputData[i]+databuffer[i];
                }
                else
                {
                    qDebug()<<QString("Sync Data Error! Port %1:").arg(i)
                           <<databuffer[i].front()->timestamp
                          <<inputData[i].back()->timestamp;
                }
            }
            else
            {
                paramsbuffer[i]=inputParams[i];
                databuffer[i]=inputData[i];
            }
        }
    }
    return generateSyncData();
}

bool Sync::generateSyncData()
{
    if(databuffer[baseportid].size()==0)
    {
        return 0;
    }
    uint i,n=databuffer.size();
    QTime basetimestamp=databuffer[baseportid].back()->timestamp;
    if(basetimestamp.isNull())
    {
        return 0;
    }
    for(i=syncrecordid;i<n;i++)
    {
        if(i==baseportid)
        {
            syncparams[i]=paramsbuffer[i].back();
            syncdata[i]=databuffer[i].back();
            deltatime[i]=0;
            syncrecordid=i+1;
            continue;
        }
        int j,m=databuffer[i].size();
        for(j=m-1;j>=0;j--)
        {
            QTime targettimestamp=databuffer[i].at(j)->timestamp;
            int delta=basetimestamp.msecsTo(targettimestamp);
            if(delta>=0)
            {
                syncparams[i]=paramsbuffer[i].back();
                syncdata[i]=databuffer[i].back();
                deltatime[i]=delta;
                syncrecordid=i+1;
                break;
            }
            else if(j>0)
            {
                QTime nexttargettimestamp=databuffer[i].at(j-1)->timestamp;
                int nextdelta=basetimestamp.msecsTo(nexttargettimestamp);
                if(nextdelta<0||(nextdelta<-delta))
                {
                    paramsbuffer[i].pop_back();
                    databuffer[i].pop_back();
                }
                else
                {
                    syncparams[i]=paramsbuffer[i].back();
                    syncdata[i]=databuffer[i].back();
                    deltatime[i]=delta;
                    syncrecordid=i+1;
                    break;
                }
            }
        }
        if(j<0)
        {
            return 0;
        }

    }
    paramsbuffer[baseportid].pop_back();
    databuffer[baseportid].pop_back();
    syncrecordid=0;
    return 1;
}

TRANSFER_PORT_PARAMS_TYPE Sync::getParams(uint portID)
{
    return syncparams[portID];
}

TRANSFER_PORT_DATA_TYPE Sync::getData(uint portID)
{
    return syncdata[portID];
}

void Sync::clear()
{
    uint i,n=databuffer.size();
    for(i=0;i<n;i++)
    {
        paramsbuffer[i].clear();
        databuffer[i].clear();
    }
    syncrecordid=0;
}
