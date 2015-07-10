TEMPLATE = subdirs

SUBDIRS += \
    Camera \
    Velodyne \
    VirtualScan \
    Localization \
    DPM \
    Fusion

DPM.depends += Camera
VirtualScan.depends += Velodyne Localization DPM Camera
Fusion.depends += Camera Velodyne VirtualScan DPM
