#include "storage.h"

StorageMono::StorageMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: DrainMono(qstrSharedLibrary,QString("StorageMono"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
}

StorageMulti::StorageMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: DrainMulti(qstrSharedLibrary,QString("StorageMulti"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
}
