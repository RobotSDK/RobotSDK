TEMPLATE = subdirs

SUBDIRS += \
    Camera \
    Velodyne \
    VirtualScan \
    Localization \
    DPM \
    Fusion \
    ObstacleMap

DPM.depends += Camera
VirtualScan.depends += Velodyne
Fusion.depends += Camera Velodyne VirtualScan DPM
