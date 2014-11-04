TEMPLATE = subdirs

unix{
SUBDIRS += \
    ConfigSystem \
    ConfigModule \
    ConfigProject \
#   ConfigApplication \
#   Publish
}

win32{
SUBDIRS += \
    ConfigSystem \
    ConfigModule \
    ConfigProject \
#   ConfigApplication \
#   Publish
}

