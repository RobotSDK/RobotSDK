#ifndef STORAGE_H
#define STORAGE_H

/*! \defgroup StorageMono_Library StorageMono_Library
	\ingroup DrainMono_Library
	\brief The Library of StorageMono.
*/

/*! \defgroup StorageMulti_Library StorageMulti_Library
	\ingroup DrainMulti_Library
	\brief The Library of StorageMulti.
*/

/*! \addtogroup ExDrain
	@{
*/

/*! \file storage.h
	\brief Defines the class StorageMono and StorageMulti.
*/

#include<Modules/Drain/drain.h>

/*! \class StorageMono
	\brief StorageMono stores mono drain data.
	\details
	Requirement
	- \ref InputPort::inputbuffersize<=0
	- \ref InputPort::grabInputData(int grabSize) grabSize==0.
*/
class StorageMono : public DrainMono
{
	Q_OBJECT
public:
	/*! \fn StorageMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class DrainMono. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMono::processMonoDrainData.
		\details
		- Set the type-name as "StorageMono"
	*/
	StorageMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
};

/*! \class StorageMulti
	\brief StorageMulti stores multi drain data.
	\details
	Requirement
	- \ref InputPort::inputbuffersize<=0
	- \ref InputPort::grabInputData(int grabSize) grabSize==0.
*/
class StorageMulti : public DrainMulti
{
	Q_OBJECT
public:
	/*! \fn StorageMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class DrainMono. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMulti::processMultiDrainData.
		\details
		- Set the type-name as "StorageMulti"
	*/
	StorageMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
};

/*! @}*/

#endif // STORAGE_H
