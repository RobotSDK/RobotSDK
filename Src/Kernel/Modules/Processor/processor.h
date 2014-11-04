#ifndef PROCESSOR_H
#define PROCESSOR_H

/*! \defgroup ProcessorMono_Library ProcessorMono_Library
	\ingroup Node_Library
	\brief The Library of ProcessorMono.
*/

/*! \defgroup ProcessorMulti_Library ProcessorMulti_Library
	\ingroup Node_Library
	\brief The Library of ProcessorMulti.
*/

/*! \defgroup Processor Processor
	\ingroup Modules
	\brief The Processor node.
*/

/*! \defgroup ExProcessor ExProcessor
	\ingroup ExModules
	\brief The ExProcessor node.
*/

/*! \addtogroup Processor 
	@{
*/

/*! \file processor.h
	\brief Defines the Processor node.
	\details A node can process input data and generate output data.
*/

#include<Core/Node/node.h>

/*!	\def PROCESSORSLOT
	\brief MACRO of Source::processInputDataSlot() to receive InputPort::inputDataSignal() or trigger signal.
*/
#ifndef PROCESSORSLOT
#define PROCESSORSLOT SLOT(processInputDataSlot())
#endif

/*! \class Processor
	\brief Processor is one of the four basic modules. It can process input data and generate output data.
	\details
	Processor: \image html Processor.png
	There are two kinds of derivative modules:
	- ProcessorMono
	- ProcessorMulti

	Remarks:
	- Processor cannot be used directly in the application.
	- Provides 1 interface function:
		- [private] Processor::initializeOutputData
	- Provides 1 set of Qt Signal-Slot:
		-Process input data
			- [slot] pure virtual Processor::processInputDataSlot();
			- [signal] Processor::processInputDataSignal();
			- [signal] Processor::processInputDataErrorSignal();
*/
class Processor : public Node
{
	Q_OBJECT
public:
	/*! \fn Processor(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName)
		\brief The constructor of the class Processor.
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\details
		- Check output ports.
	*/
	Processor(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName);
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
protected:
	/*! \var initializeOutputData
		\brief [private] Interface function of initializing node's output data.
	*/
	initializeOutputDataFptr initializeOutputData;
public slots:
	/*! \fn pure virtual void processInputDataSlot()
		\brief The slot function for processing input data.
	*/
	virtual void processInputDataSlot()=0;
signals:
	/*! \fn void processInputDataSignal();
		\brief The signal function for processing input data normally.
	*/
	void processInputDataSignal();
	/*! \fn void processInputDataErrorSignal();
		\brief The signal function for processing input data with error.
	*/
	void processInputDataErrorSignal();
};

/*! \class ProcessorMono
	\brief ProcessorMono is derived from Processor. It can process mono input data and generate output data.
	\details
	There is no extended module:

	Remarks:
	- ProcessorMono can be used directly in the application.
	- Provides 2 interface function:
		- [private] ProcessorMono::getMonoInputDataSize
		- [private] ProcessorMono::processMonoInputData
	- Overload [slot] ProcessorMono::processInputDataSlot();
*/
class ProcessorMono : public Processor
{
	Q_OBJECT
public:
	/*! \fn ProcessorMono(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
		\brief The constructor of the class ProcessorMono. (For extended modules)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of ProcessorMono::processMonoInputData.
		\details
		- Load and check the shared library.
		- Check input ports.
	*/
	ProcessorMono(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx);
	/*! \fn ProcessorMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class ProcessorMono. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of ProcessorMono::processMonoInputData.
		\details
		- Set the type-name as "ProcessorMono"
		- Load and check the shared library.
		- Check input ports.
	*/
	ProcessorMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
protected:
	/*! \typedef void (*getMonoInputDataSizeFptr)(void * paramsPtr, void * varsPtr, int & inputDataSize)
		\brief [required] Function pointer type for interface function of getting mono input data size.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [out] inputDataSize The required mono input data size. (see InputNode::grabInputData(int grabSize))
	*/
	typedef void (*getMonoInputDataSizeFptr)(void * paramsPtr, void * varsPtr, int & inputDataSize);
	/*! \typedef bool (*processMonoInputDataFptr)(void * paramsPtr, void * varsPtr, QVector<void *> inputParams, QVector<void *> inputData, void * outputData, QList<int> & outputPortIndex)
		\brief [required] Function pointer type for interface function of processing mono input data.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [in] inputParams The mono input parameters of input node.
		\param [in] inputData The mono input data of input node.
		\param [out] outputData The output data.
		\param [out] outputPortIndex The index of output port to send output data.
		\return 1 for success and 0 for failure.
	*/
	typedef bool (*processMonoInputDataFptr)(void * paramsPtr, void * varsPtr, QVector<void *> inputParams, QVector<void *> inputData, void * outputData, QList<int> & outputPortIndex);
protected:
	/*! \var getMonoInputDataSize
		\brief [private] Interface function of getting mono input data size.
	*/
	getMonoInputDataSizeFptr getMonoInputDataSize;
	/*! \var processMonoInputData
		\brief [private] Interface function of processing mono input data.
	*/
	processMonoInputDataFptr processMonoInputData;
public slots:
	/*! \fn void processInputDataSlot()
		\brief The slot function for processing mono input data.
	*/
	void processInputDataSlot();
};

/*! \class ProcessorMulti
	\brief ProcessorMulti is derived from Processor. It can process multi input data and generate output data.
	\details
	There is no extended module:

	Remarks:
	- ProcessorMulti can be used directly in the application.
	- Provides 2 interface function:
		- [private] ProcessorMulti::getMultiInputDataSize
		- [private] ProcessorMulti::processMultiInputData
	- Overload [slot] ProcessorMulti::processInputDataSlot();
*/
class ProcessorMulti : public Processor
{
	Q_OBJECT
public:
	/*! \fn ProcessorMulti(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
		\brief The constructor of the class ProcessorMulti. (For extended modules)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeType The type-name of the node.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of ProcessorMulti::processMultiInputData.
		\details
		- Load and check the shared library.
		- Check input ports.
	*/
	ProcessorMulti(QString qstrSharedLibrary, QString qstrNodeType, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx);
	/*! \fn ProcessorMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class ProcessorMulti. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of ProcessorMulti::processMultiInputData.
		\details
		- Set the type-name as "ProcessorMulti"
		- Load and check the shared library.
		- Check input ports.
	*/
	ProcessorMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
protected:
	/*! \typedef void (*getMultiInputDataSizeFptr)(void * paramsPtr, void * varsPtr, QList<int> & inputDataSize)
		\brief [required] Function pointer type for interface function of getting multi input data size.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [out] inputDataSize The required multi input data size. (see InputNode::grabInputData(int grabSize))
	*/
	typedef void (*getMultiInputDataSizeFptr)(void * paramsPtr, void * varsPtr, QList<int> & inputDataSize);
    /*! \typedef bool (*processMultiInputDataFptr)(void * paramsPtr, void * varsPtr, QVector<QVector<void *> > inputParams, QVector<QVector<void *> > inputData, void * outputData, QList<int> & outputPortIndex);
		\brief [required] Function pointer type for interface function of processing multi input data.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr).
		\param [in] inputParams The multi input parameters of input node.
		\param [in] inputData The multi input data of input node.
		\param [out] outputData The output data.
		\param [out] outputPortIndex The index of output port to send output data.
		\return 1 for success and 0 for failure.
	*/
    typedef bool (*processMultiInputDataFptr)(void * paramsPtr, void * varsPtr, QVector<QVector<void *> > inputParams, QVector<QVector<void *> > inputData, void * outputData, QList<int> & outputPortIndex);
protected:
	/*! \var getMultiInputDataSize
		\brief [private] Interface function of getting multi input data size.
	*/
	getMultiInputDataSizeFptr getMultiInputDataSize;
	/*! \var processMultiInputData
		\brief [private] Interface function of processing multi input data.
	*/
	processMultiInputDataFptr processMultiInputData;
public slots:
	/*! \fn void processInputDataSlot()
		\brief The slot function for processing multi input data.
	*/
	void processInputDataSlot();
};

/*! @} */

#endif // PROCESSOR_H
