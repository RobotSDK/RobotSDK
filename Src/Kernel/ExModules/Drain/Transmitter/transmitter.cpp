#include "transmitter.h"

TransmitterMono::TransmitterMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: DrainMono(qstrSharedLibrary,QString("TransmitterMono"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
}

TransmitterMulti::TransmitterMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: DrainMulti(qstrSharedLibrary,QString("TransmitterMulti"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
}
