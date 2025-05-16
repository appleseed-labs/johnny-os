# Code to control the Linak LA36 actuator using a CAN-USB interface
We use a kvaser leaf2 usb to can to connect to the actuator. The can protocol is **canopen**. We basically followed the [protocol user manual](https://cdn.linak.com/-/media/files/ic-and-bus-actuators/techline-canopen-user-manual-eng-legacy.pdf?_gl=1*1os9jmo*_gcl_au*NzcwNjk5ODIyLjE3NDQyOTk4MjQ.)  and [connection diagrams](https://cdn.linak.com/-/media/files/connection-diagrams/la36/la36-37--16-canopen.pdf) from Linak.

## Initialize CAN using command line
The commands below intialize the can bus:
```
sudo modprobe can
sudo modprobe can_raw
sudo ip link set can1 type can bitrate 125000
sudo ip link set up can1

```
**Note** If there is another kvaser device connected there should be interfaces can0 and can1. Double check what corresponds to the linear actuator and adapt the lines above accordingly.

## Initialize CAN as service
The commands to initialize CAN (noted above) is run as a service everytime the system boots up. The script with the commands is at `/etc/systemd/system/initializeCAN.sh` and the service script is at `/etc/systemd/system/initializeCAN.service`.

Create a new file at /etc/systemd/system/initializeCAN.service with the following content:
```
[Unit]
Description=Initialize CAN interface
After=network.target

[Service]
ExecStart=/etc/systemd/system/initializeCAN.sh
RemainAfterExit=true
Type=oneshot
User=root

[Install]
WantedBy=multi-user.target
```
Enable the Service to Start at Boot:
```
sudo systemctl enable initializeCAN.service
```
To start the service immediately (without rebooting), use the following command:
```
sudo systemctl start initializeCAN.service
```

You can check the status of the service by
```
sudo systemctl status initializeCAN.service
```
and restart the service at any point using the command
```
sudo systemctl restart initializeCAN.service
```

To run the node do:
```
ros2 run linak_controller linak_control
```
The node subscribes to the **/hole_drilling** topic. Any message of the type **std_msgs/msg/String** there will make the actuator to extend. After 5 secs it will automatically retract.
