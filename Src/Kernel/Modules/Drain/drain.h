#ifndef DRAIN_H
#define DRAIN_H

/*! \defgroup DrainMono_Library DrainMono_Library
	\ingroup Node_Library
	\brief The Library of DrainMono.
*/

/*! \defgroup DrainMulti_Library DrainMulti_Library
	\ingroup Node_Library
	\brief The Library of DrainMulti.
*/

/*! \defgroup Drain Drain
	\ingroup Modules
	\brief The Drain node.
*/

/*! \defgroup ExDrain ExDrain
	\ingroup ExModules
	\brief The ExDrain node.
*/

/*! \addtogroup Drain 
	@{
*/

/*! \file drain.h
	\brief Defines the Drain %Node.
	\details A node can only process drain data.
*/

#include<Core/Node/node.h>

/*!	\def DRAINSLOT
	\brief MACRO of Drain::processDrainDataSlot() or SourceDrain::processDrainDataSlot() to receive InputPort::drainDataSignal() or trigger signal.
*/
#ifndef DRAINSLOT
#define DRAINSLOT SLOT(processDrainDataSlot())
#endif

/*! \class Drain
	\brief Drain is one of the four basic modules. It can only process drain data.
	\details
	Drain: \image html Drain.png
	There are two kinds of derivative modules:
	- DrainMono
	- DrainMulti

	Remarks:
	- Drain cannot be used directly in the application.
	- Provides 1 set of Qt Signal-Slot:
		-Process drain data
			- [slot] pure virtual Drain::processDrainDataSlot();
			- [signal] Drain::processDrainDataSignal();
			- [signal] Drain::processDrainDataErrorSignal();
*/
class Drain : public Node
{
	Q_OBJECT
public:
	/*! \fn Drain(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName)
		\brief The constructor of the class Drain.
		\param [in] qstrSharedLibrary The name of the shared library.		
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\details
		- Check output ports.
	*/
	Drain(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName);
public slots:
	/*! \fn pure virtual void processDrainDataSlot()
		\brief The slot function for processing drain data.
	*/
	virtual void processDrainDataSlot()=0;
signals:
	/*! \fn void processDrainDataSignal();
		\brief The signal function for processing drain data normally.
	*/
	void processDrainDataSignal();
	/*! \fn void processDrainDataErrorSignal();
		\brief The signal function for processing drain data with error.
	*/
	void processDrainDataErrorSignal();
};

/*! \class DrainMono
	\brief DrainMono is derived from Drain. It can only process mono drain data.
	\details
	There are three kinds of extended modules: (Only the class-names are different)
	- StorageMono: Store data (File)
	- TransmitterMono: Transmit data (UDP, COMPORT)
	- VisualizationMono: Visualize drain data.

	Remarks:
	- DrainMono can be used directly in the application.
	- Provides 2 interface function:
		- [private] DrainMono::getMonoDrainDataSize
		- [private] DrainMono::processMonoDrainData
	- Overload [slot] DrainMono::processDrainDataSlot();
*/
class DrainMono : public Drain
{
	Q_OBJECT
public:
	/*! \fn DrainMono(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class DrainMono. (For extended modules)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMono::processMonoDrainData.
		\details
		- Load and check the shared library.
		- Check input ports.
	*/
	DrainMono(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
	/*! \fn DrainMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class DrainMono. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.	
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMono::processMonoDrainData.
		\details
		- Set the type-name as "DrainMono"
		- Load and check the shared library.
		- Check input ports.
	*/
	DrainMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
protected:
	/*! \typedef void (*getMonoDrainDataSizeFptr)(void * paramsPtr, void * varsPtr, int & drainDataSize)
		\brief [required] Function pointer type for interface function of getting mono drain data size.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [out] drainDataSize The required mono drain data size. (see InputNode::grabDrainData(int grabSize))
	*/
	typedef void (*getMonoDrainDataSizeFptr)(void * paramsPtr, void * varsPtr, int & drainDataSize);
	/*! \typedef bool (*processMonoDrainDataFptr)(void * paramsPtr, void * varsPtr, QVector<void *> drainParams, QVector<void *> drainData)
		\brief [required] Function pointer type for interface function of processing mono drain data.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [in] drainParams The mono drain parameters of input node.
		\param [in] drainData The mono drain data of input node.
	*/
	typedef bool (*processMonoDrainDataFptr)(void * paramsPtr, void * varsPtr, QVector<void *> drainParams, QVector<void *> drainData);
protected:
	/*! \var getMonoDrainDataSize
		\brief [private] Interface function of getting mono drain data size.
	*/
	getMonoDrainDataSizeFptr getMonoDrainDataSize;
	/*! \var processMonoDrainData
		\brief [private] Interface function of processing mono drain data.
	*/
	processMonoDrainDataFptr processMonoDrainData;
public slots:
	/*! \fn void processDrainDataSlot()
		\brief The slot function for processing mono drain data.
	*/
	void processDrainDataSlot();
};

/*! \class DrainMulti
	\brief DrainMulti is derived from Drain. It can only process multi drain data.
	\details
	There are three kinds of extended modules: (Only the class-names are different)
	- StorageMulti: Store data (File)
	- TransmitterMulti: Transmit data (UDP, COMPORT)
	- VisualizationMulti: Visualize drain data.

	Remarks:
	- DrainMulti can be used directly in the application.
	- Provides 2 interface function:
		- [private] DrainMulti::getMultiDrainDataSize
		- [private] DrainMulti::processMultiDrainData
	- Overload [slot] DrainMulti::processDrainDataSlot();
*/
class DrainMulti : public Drain
{
	Q_OBJECT
public:
	/*! \fn DrainMulti(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
		\brief The constructor of the class DrainMulti. (For extended modules)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMulti::processMultiDrainData.
		\details
		- Load and check the shared library.
		- Check input ports.
	*/
	DrainMulti(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx);
	/*! \fn DrainMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class DrainMulti. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMulti::processMultiDrainData.
		\details
		- Set the type-name as "DrainMulti"
		- Load and check the shared library.
		- Check input ports.
	*/
	DrainMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
protected:
	/*! \typedef void (*getMultiDrainDataSizeFptr)(void * paramsPtr, void * varsPtr, QList<int> & drainDataSize)
		\brief [required] Function pointer type for interface function of getting multi drain data size.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [out] drainDataSize The required multi drain data size. (see InputNode::grabDrainData(int grabSize))
	*/
	typedef void (*getMultiDrainDataSizeFptr)(void * paramsPtr, void * varsPtr, QList<int> & drainDataSize);
    /*! \typedef bool (*processMultiDrainDataFptr)(void * paramsPtr, void * varsPtr,  QVector<QVector<void *> > drainParams, QVector<QVector<void *> > drainData);
		\brief [required] Function pointer type for interface function of processing multi drain data.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [in] drainParams The multi drain parameters of input node.
		\param [in] drainData The multi drain data of input node.
	*/
    typedef bool (*processMultiDrainDataFptr)(void * paramsPtr, void * varsPtr, QVector<QVector<void *> > drainParams, QVector<QVector<void *> > drainData);
protected:
	/*! \var getMultiDrainDataSize
		\brief [private] Interface function of getting multi drain data size.
	*/
	getMultiDrainDataSizeFptr getMultiDrainDataSize;
	/*! \var processMultiDrainData
		\brief [private] Interface function of processing multi drain data.
	*/
	processMultiDrainDataFptr processMultiDrainData;
public slots:
	/*! \fn void processDrainDataSlot()
		\brief The slot function for processing multi drain data.
	*/
	void processDrainDataSlot();
};

/*! @}*/

#endif // DRAIN_H
