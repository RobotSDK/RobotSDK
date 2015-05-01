#ifndef SYNC_H
#define SYNC_H

#include<valuebase.h>

namespace RobotSDK
{

class Sync
{
public:
    Sync(uint portNum, uint basePortID);
protected:
    uint baseportid;
    uint syncrecordid;
    QVector< QList < TRANSFER_PORT_PARAMS_TYPE > > paramsbuffer;
    QVector< QList < TRANSFER_PORT_DATA_TYPE > > databuffer;
    QVector< TRANSFER_PORT_PARAMS_TYPE > syncparams;
    QVector< TRANSFER_PORT_DATA_TYPE > syncdata;
    QVector< int > deltatime;
protected:
    bool generateSyncData();
public:
    bool addParamsData(PORT_PARAMS_CAPSULE & inputParams, PORT_DATA_CAPSULE & inputData);
    TRANSFER_PORT_PARAMS_TYPE getParams(uint portID);
    TRANSFER_PORT_DATA_TYPE getData(uint portID);
    void clear();
};


#define ADD_SYNC(syncName, basePortID) public: RobotSDK::Sync syncName=RobotSDK::Sync(INPUT_PORT_NUM, basePortID);

#define SYNC_START(sync) IS_INTERNAL_TRIGGER? \
    false : sync.addParamsData(INPUT_PARAMS_ARG, INPUT_DATA_ARG)

#define SYNC_PARAMS(sync, portID) (portID>=0 && portID<INPUT_PORT_NUM && portID<INPUT_PARAMS_ARG.size()) ? \
    std::static_pointer_cast< const PORT_PARAMS_TYPE(portID) >(sync.getParams(portID)) \
  : std::shared_ptr< const PORT_PARAMS_TYPE(portID) >()

#define SYNC_DATA(sync, portID) (portID>=0 && portID<INPUT_PORT_NUM && portID<INPUT_DATA_ARG.size()) ? \
    std::static_pointer_cast< const PORT_DATA_TYPE(portID) >(sync.getData(portID)) \
  : std::shared_ptr< const PORT_DATA_TYPE(portID) >()

#define SYNC_CLEAR(sync) sync.clear();

}

#endif // SYNC_H
