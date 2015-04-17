#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsView>

#include<xgraph.h>

namespace Ui {
class MainWindow;
}

class GraphView : public QGraphicsView
{
    Q_OBJECT
public:
    GraphView(QWidget * parent=0);
signals:
    void signalHandleMenu();
protected:
    double ratio=1;
    bool ctrlflag=0;
protected:
    void mousePressEvent(QMouseEvent * event);
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

protected:
    GraphView * view;
    XGraph * graph;
};

#endif // MAINWINDOW_H
