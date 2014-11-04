#ifndef TRANSMITTER_H
#define TRANSMITTER_H

/*! \defgroup TransmitterMono_Library TransmitterMono_Library
	\ingroup DrainMono_Library
	\brief The Library of TransmitterMono.
*/

/*! \defgroup TransmitterMulti_Library TransmitterMulti_Library
	\ingroup DrainMulti_Library
	\brief The Library of TransmitterMulti.
*/

/*! \addtogroup ExDrain
	@{
*/

/*! \file transmitter.h
	\brief Defines the class TransmitterMono and TransmitterMulti.
*/

#include<Modules/Drain/drain.h>

/*! \class TransmitterMono
	\brief TransmitterMono transmitter mono drain data.
	\details
	Requirement
	- \ref InputPort::inputbuffersize<=0
*/
class TransmitterMono : public DrainMono
{
	Q_OBJECT
public:
	/*! \fn TransmitterMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class TransmitterMono. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMono::processMonoDrainData.s
		\details
		- Set the type-name as "TransmitterMono"
	*/
	TransmitterMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
};

/*! \class TransmitterMulti
	\brief TransmitterMulti transmitters multi drain data.
	\details
	Requirement
	- \ref InputPort::inputbuffersize<=0
*/
class TransmitterMulti : public DrainMulti
{
	Q_OBJECT
public:
	/*! \fn TransmitterMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class TransmitterMulti. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMulti::processMultiDrainData.
		\details
		- Set the type-name as "TransmitterMulti"
	*/
	TransmitterMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
};

/*! @}*/

#endif // TRANSMITTER_H
