#ifndef REGISTERDIRWIDGETITEM_H
#define REGISTERDIRWIDGETITEM_H

#include <QWidget>
#include "ui_registerdirwidgetitem.h"

#include<qfile.h>
#include<qtextstream.h>
#include<qdom.h>
#include<qstring.h>
#include<qtablewidget.h>
#include<qdir.h>
#include<qmessagebox.h>
#include<qfiledialog.h>
#include<qinputdialog.h>
#include<stdlib.h>

#ifdef Q_OS_LINUX

#include<qglobal.h>

#define PATHATTR "PATH"
#define ROBOTSDKSH QString("%1/SDK/RobotSDK/Tools/RobotSDK.sh").arg(QString(qgetenv("HOME")))
#define ROBOTSDKPATHSH QString("%1/SDK/RobotSDK/Tools/RobotSDK_PATH.sh").arg(QString(qgetenv("HOME")))

QString getRegValue(QString regValue);
bool setRegValue(QString regValue, QString data);
bool deleteRegValue(QString regValue);

#elif defined(Q_OS_WIN)

#include<Windows.h>

#define PATHATTR "Path"
#define REGENVPATH "SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment"
#define REGDATABUFFERSIZE 5000

QString getRegValue(HKEY hkey, QString subkey, QString regValue);
bool setRegValue(HKEY hkey, QString subkey, QString regValue, DWORD dataType, QString data);
bool deleteRegValue(HKEY hkey, QString subkey, QString regValue);

#endif

class RegisterDirWidgetItem : public QWidget
{
	Q_OBJECT
public:
    RegisterDirWidgetItem(QWidget * parent, QString qstrRegisterValue,bool editable=1);
	~RegisterDirWidgetItem();
private:
	Ui::RegisterDirWidgetItem ui;
protected:
	QDomDocument * doc;
public:
	void setQDomDocument(QDomDocument * pDoc);
	void loadRegisterDir(QDomElement & rootElem);
	void saveRegisterDir(QDomElement & rootElem);
	void registerDir();
protected:
	bool registerPath(QString registerValue,bool addFlag);
public slots:
	void addRegisterDirTagSlot();
	void deleteRegisterDirTagSlot();
	void clearRegisterDirTagSlot();
	void processDoubleClickedSlot(int row, int column);
public:
	QString getRegisterValue();
	bool hasData();
};

#endif // REGISTERDIRWIDGETITEM_H
