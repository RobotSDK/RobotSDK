TEMPLATE = subdirs

SUBDIRS += \
    Camera \
    Velodyne \
    VirtualScan \
    Localization \
    DPM \
    Fusion

DPM.depends += Camera
VirtualScan.depends += Velodyne
Fusion.depends += Camera Velodyne VirtualScan
