#ifndef SYNC_H
#define SYNC_H

#include<valuebase.h>

namespace RobotSDK
{

class Sync
{
public:
    Sync(uint portNum, uint baseID, QList<uint> specPortIDs=QList<uint>());
protected:
    uint baseid;
    QList<uint> specportids;
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
    TRANSFER_PORT_PARAMS_TYPE getParams(uint syncID);
    TRANSFER_PORT_DATA_TYPE getData(uint syncID);
    void clear();
};


#define ADD_SYNC(syncName, basePortID) public: Sync syncName=Sync(INPUT_PORT_NUM, basePortID);

#define ADD_SYNC_SPEC(syncName, specPortIDs, baseSyncID) public: Sync syncName=Sync(specPortIDs.size(), baseSyncID, specPortIDs);

#define SYNC_START(sync) IS_INTERNAL_TRIGGER? \
    false : sync.addParamsData(INPUT_PARAMS_ARG, INPUT_DATA_ARG)

#define SYNC_PARAMS(sync, syncID) (syncID>=0 && syncID<INPUT_PORT_NUM && syncID<INPUT_PARAMS_ARG.size()) ? \
    std::static_pointer_cast< const PORT_PARAMS_TYPE(syncID) >(sync.getParams(syncID)) \
  : std::shared_ptr< const PORT_PARAMS_TYPE(syncID) >()

#define SYNC_DATA(sync, syncID) (syncID>=0 && syncID<INPUT_PORT_NUM && syncID<INPUT_DATA_ARG.size()) ? \
    std::static_pointer_cast< const PORT_DATA_TYPE(syncID) >(sync.getData(syncID)) \
  : std::shared_ptr< const PORT_DATA_TYPE(syncID) >()

#define SYNC_CLEAR(sync) sync.clear();

}

#endif // SYNC_H
