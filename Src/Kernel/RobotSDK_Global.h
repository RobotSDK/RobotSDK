#ifndef ROBOTSDK_GLOBAL_H
#define ROBOTSDK_GLOBAL_H

/*! \addtogroup Kernel
	@{
*/

/*! \file RobotSDK_Global.h
	\brief RobotSDK global definition.
	\details
	- Used in shared library creation.
	- Used in application's reference to shared library's _ParamsData.h.
*/

#include<qglobal.h>

/*! \def ROBOTSDK_OUTPUT
	\brief Defines the output type according to the Macro RobotSDK_ModuleDev in the project.
	\details
	- If undefined RobotSDK_ModuleDev in the project, output type=Q_DECL_IMPORT which is for application.
	- If defined RobotSDK_ModuleDev in the project, output type=Q_DECL_EXPORT which is for shared library.
*/
#ifdef RobotSDK_ModuleDev
#define ROBOTSDK_OUTPUT Q_DECL_EXPORT
#else
#define ROBOTSDK_OUTPUT Q_DECL_IMPORT
#endif

/*! \def DECOFUNC(func)
	\brief To decorate function  func.
*/
/*! \def DECOFUNC_1(NODECONFIG,func)
	\brief To decorate function  func using NODECONFIG.
*/
/*! \def DECOFUNC_2(NODECONFIG,func)
	\brief To decorate function  func using NODECONFIG and get function name as NODECONFIG_func.
*/
#define DECOFUNC(func) DECOFUNC_1(NODECONFIG,func)
#define DECOFUNC_1(NODECONFIG,func) DECOFUNC_2(NODECONFIG,func)
#define DECOFUNC_2(NODECONFIG,func) NODECONFIG##_##func

#include<ExModules/Drain/Visualization/visualization.h>
#include<ExModules/Source/UserInput/userinput.h>
#include<ExModules/Drain/Transmitter/transmitter.h>
#include<Core/Edge/triggerlog.h>
#include<ExModules/Source/Simulator/simulator.h>
#include<ExModules/Source/Sensor/sensor.h>
#include<Core/Edge/triggerview.h>
#include<Modules/SourceDrain/sourcedrain.h>
#include<Modules/Processor/processor.h>
#include<Modules/Source/source.h>
#include<Core/Edge/edge.h>
#include<ExModules/Drain/Storage/storage.h>
#include<Core/Node/node.h>
#include<Modules/Drain/drain.h>

#include<qglobal.h>
#include<qdebug.h>
#include<qlabel.h>
#include<qlineedit.h>
#include<qstring.h>
#include<qfile.h>
#include<qlist.h>
#include<qvector.h>
#include<qset.h>
#include<qfile.h>
#include<qtextstream.h>
#include<qdatetime.h>
#include<qtimer.h>
#include<qimage.h>
#include<qpainter.h>
#include<qrgb.h>
#include<boost/shared_ptr.hpp>
#include<Accessories/XMLDomInterface/xmldominterface.h>

/*!	\fn void copyQVector(QVector<T1 *> & dst, QVector<T2 *> & src)
	\brief Copy and convert pointers.
	\param [in] dst The destination to store pointers.
	\param [in] src The source to copy.
*/
template<class T1, class T2>
void copyQVector(QVector<T1 *> & dst, QVector<T2 *> & src)
{
	int i,n=src.size();
	dst.resize(n);
	for(i=0;i<n;i++)
	{
		dst[i]=(T1 *)src[i];
	}
}

/*! @}*/ 

#endif