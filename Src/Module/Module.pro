TEMPLATE = subdirs

SUBDIRS += \
    Camera \
    Velodyne \
    VirtualScan \
    ObstacleMap \
    Localization \
    DPM \
    Fusion

DPM.depends += Camera
VirtualScan.depends += Velodyne
ObstacleMap.depends += VirtualScan
Fusion.depends += Camera Velodyne VirtualScan DPM
