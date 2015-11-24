#-------------------------------------------------
#
# Project created by QtCreator 2015-06-19T14:50:33
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = TestParticleFilter
CONFIG += console
CONFIG -= app_bundle
CONFIG += c++11

TEMPLATE = app

HEADERS += \
    particlefilter.h \
    particlefilter.cuh \
    egotransform.h \
    particlefilterdef.h \
    particlefilterbase.h \
    randomgenerator.h

SOURCES += \
    main.cpp

DISTFILES += \
    egotransform.cu \
    particlefilter.cu

include($$(ROBOTSDKCUDA))


