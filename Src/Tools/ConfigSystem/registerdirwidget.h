#ifndef REGISTERDIRWIDGET_H
#define REGISTERDIRWIDGET_H

#include <QWidget>
#include "ui_registerdirwidget.h"

#include"registerdirwidgetitem.h"

class RegisterDirWidget : public QWidget
{
	Q_OBJECT

public:
	RegisterDirWidget(QWidget * parent, QDomDocument * qDoc, QString qstrRegisterName);
    RegisterDirWidget(QWidget * parent, QDomDocument * qDoc, QString qstrRegisterName, QStringList qstrlsRegisterValue, bool editable=1);
	~RegisterDirWidget();
private:
	Ui::RegisterDirWidget ui;
protected:
	QDomDocument * doc;
public:
	void setQDomDocument(QDomDocument * pDoc);
    void addItem(QString itemName,bool editable=1);
	void loadRegisterDir(QDomElement & rootElem);
	void saveRegisterDir(QDomElement & rootElem);
public slots:
	void registerDir();
public:
	QStringList getRegisterValues();
	bool hasData(int id);
};

#endif // REGISTERDIRWIDGET_H
