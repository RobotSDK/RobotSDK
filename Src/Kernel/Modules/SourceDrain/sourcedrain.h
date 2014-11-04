#ifndef SOURCEDRAIN_H
#define SOURCEDRAIN_H

/*! \defgroup SourceDrainMono_Library SourceDrainMono_Library
	\ingroup Node_Library
	\brief The Library of SourceDrainMono.
*/

/*! \defgroup SourceDrainMulti_Library SourceDrainMulti_Library
	\ingroup Node_Library
	\brief The Library of SourceDrainMulti.
*/

/*! \defgroup SourceDrain SourceDrain
	\ingroup Modules
	\brief The SourceDrain node.
*/

/*! \defgroup ExSourceDrain ExSourceDrain
	\ingroup ExModules
	\brief The ExSourceDrain node.
*/

/*! \addtogroup SourceDrain 
	@{
*/

/*! \file sourcedrain.h
	\brief Defines the SourceDrain node.
	\details A node can process drain data and generate source data.
*/

#include<Core/Node/node.h>

/*!	\def SOURCESLOT
	\brief MACRO of Source::generateSourceDataSlot() or SourceDrain::generateSourceDataSlot() to receive trigger signal.
*/
#ifndef SOURCESLOT
#define SOURCESLOT SLOT(generateSourceDataSlot())
#endif

/*!	\def DRAINSLOT
	\brief MACRO of Drain::processDrainDataSlot() or SourceDrain::processDrainDataSlot() to receive InputPort::drainDataSignal() or trigger signal.
*/
#ifndef DRAINSLOT
#define DRAINSLOT SLOT(processDrainDataSlot())
#endif

/*! \class SourceDrain
	\brief SourceDrain is one of the four basic modules. It can process drain data and generate source data.
	\details
	SourceDrain: \image html Source-Drain.png
	There are two kinds of derivative modules:
	- SourceDrainMono
	- SourceDrainMulti

	Remarks:
	- SourceDrain cannot be used directly in the application.
	- Provides 2 interface function:
		- [private] SourceDrain::initializeOutputData
		- [private] SourceDrain::generateSourceData
	- Provides 2 set of Qt Signal-Slot:
		-Generate source data
			- [slot] virtual SourceDrain::generateSourceDataSlot();
			- [signal] SourceDrain::generateSourceDataSignal();
			- [signal] SourceDrain::generateSourceDataErrorSignal();
		-Process drain data
			- [slot] pure virtual SourceDrain::processDrainDataSlot();
			- [signal] SourceDrain::processDrainDataSignal();
			- [signal] SourceDrain::processDrainDataErrorSignal();
*/
class SourceDrain : public Node
{
	Q_OBJECT
public:
	/*! \fn SourceDrain(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
		\brief The constructor of the class SourceDrain.
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of SourceDrain::generateSourceDatas.
		\details
		- Check output ports.
	*/
	SourceDrain(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx);
protected:
	/*! \typedef void (*initializeOutputDataFptr)(void * paramsPtr, void * varsPtr, boost::shared_ptr<void> & outputDataPtr)
		\brief [required] Function pointer type for interface function of initializing node's output data.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [out] outputDataPtr The output data embelished by boost::shared_pointer<void>.
		\details To initialize output data:
		- outputDataPtr=boost::shared_ptr<void> (new OutputDataType);
	*/
	typedef void (*initializeOutputDataFptr)(void * paramsPtr, void * varsPtr, boost::shared_ptr<void> & outputDataPtr);
	/*! \typedef bool (*generateSourceDataFptr)(void * paramsPtr, void * varsPtr, void * outputData, QList<int> & outputPortIndex, QTime & timeStamp)
		\brief [required] Function pointer type for interface function of generating source data.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [out] outputData The output data.
		\param [out] outputPortIndex The index of output port to send output data.
		\param [out] timeStamp The timestamp. Mainly used in simulator to get the timestamp of one data frame.
		\return 1 for success and 0 for failure.
	*/
	typedef bool (*generateSourceDataFptr)(void * paramsPtr, void * varsPtr, void * outputData, QList<int> & outputPortIndex, QTime & timeStamp);
protected:
	/*! \var initializeOutputData
		\brief [private] Interface function of initializing node's output data.
	*/
	initializeOutputDataFptr initializeOutputData;
	/*! \var generateSourceData
		\brief [private] Interface function of generating source data.
	*/
	generateSourceDataFptr generateSourceData;
public slots:
	/*! \fn virtual void generateSourceDataSlot()
		\brief The slot function for generating source data.
	*/
	virtual void generateSourceDataSlot();
	/*! \fn pure virtual void processDrainDataSlot()
		\brief The slot function for processing drain data.
	*/
	virtual void processDrainDataSlot()=0;
signals:
	/*! \fn void generateSourceDataSignal();
		\brief The signal function for generating source data normally.
	*/
	void generateSourceDataSignal();
	/*! \fn void generateSourceDataErrorSignal();
		\brief The signal function for generating source data with error.
	*/
	void generateSourceDataErrorSignal();
	/*! \fn void processDrainDataSignal();
		\brief The signal function for processing drain data normally.
	*/
	void processDrainDataSignal();
	/*! \fn void processDrainDataErrorSignal();
		\brief The signal function for processing drain data with error.
	*/
	void processDrainDataErrorSignal();
};

/*! \class SourceDrainMono
	\brief SourceDrainMono is derived from SourceDrain. It can process mono drain data and generate source data.
	\details
	There is no extended module:

	Remarks:
	- SourceDrainMono can be used directly in the application.
	- Provides 2 interface function:
		- [private] SourceDrainMono::getMonoDrainDataSize
		- [private] SourceDrainMono::processMonoDrainData
	- Overload [slot] SourceDrainMono::processDrainDataSlot();
*/
class SourceDrainMono : public SourceDrain
{
	Q_OBJECT
public:
	/*! \fn SourceDrainMono(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
		\brief The constructor of the class SourceDrainMono. (For extended modules)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of SourceDrainMono::processMonoDrainData.
		\details
		- Load and check the shared library.
		- Check input ports.
	*/
	SourceDrainMono(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx);
	/*! \fn SourceDrainMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class SourceDrainMono. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of SourceDrainMono::processMonoDrainData.
		\details
		- Set the type-name as "SourceDrainMono"
		- Load and check the shared library.
		- Check input ports.
	*/
	SourceDrainMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
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
		\return 1 for success and 0 for failure.
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

/*! \class SourceDrainMulti
	\brief SourceDrainMulti is derived from SourceDrain. It can process multi drain data and generate source data.
	\details
	There is no extended module:

	Remarks:
	- SourceDrainMulti can be used directly in the application.
	- Provides 2 interface function:
		- [private] SourceDrainMulti::getMultiDrainDataSize
		- [private] SourceDrainMulti::processMultiDrainData
	- Overload [slot] SourceDrainMulti::processDrainDataSlot();
*/
class SourceDrainMulti : public SourceDrain
{
	Q_OBJECT
public:
	/*! \fn SourceDrainMulti(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
		\brief The constructor of the class SourceDrainMulti. (For extended modules)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of SourceDrainMulti::processMultiDrainData.
		\details
		- Load and check the shared library.
		- Check input ports.
	*/
	SourceDrainMulti(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx);
	/*! \fn SourceDrainMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class SourceDrainMulti. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of SourceDrainMulti::processMultiDrainData.
		\details
		- Set the type-name as "SourceDrainMulti"
		- Load and check the shared library.
		- Check input ports.
	*/
	SourceDrainMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
protected:
	/*! \typedef void (*getMultiDrainDataSizeFptr)(void * paramsPtr, void * varsPtr, QList<int> & drainDataSize)
		\brief [required] Function pointer type for interface function of getting multi drain data size.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [out] drainDataSize The required multi drain data size. (see InputNode::grabDrainData(int grabSize))
	*/
	typedef void (*getMultiDrainDataSizeFptr)(void * paramsPtr, void * varsPtr, QList<int> & drainDataSize);
    /*! \typedef bool (*processMultiDrainDataFptr)(void * paramsPtr, void * varsPtr, QVector<QVector<void *> > drainParams, QVector<QVector<void *> > drainData);
		\brief [required] Function pointer type for interface function of processing multi drain data.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [in] drainParams The multi drain parameters of input node.
		\param [in] drainData The multi drain data of input node.
		\return 1 for success and 0 for failure.
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

/*! @} */

#endif // SOURCEDRAIN_H
