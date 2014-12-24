#!/bin/sh
TMPPATH=$PWD;
mkdir -p $TMPPATH/TMP;cd $TMPPATH/TMP;
for profile in $(find $TMPPATH -type f -name "*.pro"); do
  qmake -makefile $profile "CONFIG+=release";
  make;make install;
  qmake -makefile $profile "CONFIG+=debug";
  make;make install;
done
cd $TMPPATH;
rm -rf $TMPPATH/TMP;
