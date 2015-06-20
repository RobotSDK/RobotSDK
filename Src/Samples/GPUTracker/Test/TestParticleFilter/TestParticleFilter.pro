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
    def.h \
    randomgenerator.h \
    egotransform.h \
    particlefilterbase.h \
    particlefilter.h

SOURCES += \
    main.cpp

DISTFILES += \
    randomgenerator.cu \
    egotransform.cu \
    particlefilterbase.cu \
    particlefilter.cu

include($$(ROBOTSDKCUDA))


