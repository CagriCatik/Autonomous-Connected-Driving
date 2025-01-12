# ROS Foundations

![ROS1](https://img.shields.io/badge/ROS1-blue)

Now, that you know how to start and build the ROS workspace, we can continue to learn more about the fundamental features of ROS. ROS offers many tool and helper functions that can be helpful during the development of new features. 

In this exercise you will learn
* how to use the ROS filesystem to search and navigate through the workspace
* what a ROS Master is used for and how to start it 
* what ROS Nodes are and how to launch them
* what ROS Parameters are and how to set them
* what ROS Messages are and what information they can contain
* what the difference between ROS Nodes and ROS Services
* what ROS Launch files are used for and how they are structured 

## ROS Filesystem

:warning: *Make sure that you started the `acdc` docker container and sourced the `devel/setup.bash` file in each terminal tab.* :warning:

Find the location of individual ROS packages within your file system, e.g.
```bash
rospack find racing
```
ROS commands also offer auto-completion using the Tab key. Try:
```bash
rospack find rac<<< HIT TAB KEY FOR AUTO-COMPLETE --> racing >>>
```
To navigate to individual packages, you can avoid going through many layers of subdirectories, which may be necessary when using the bash command `cd`. Instead, with `roscd`, you can immediately jump to the package location. Please execute:
```bash
roscd racing
``` 
For getting an overview of the content of a ROS package, you can use `rosls` instead of navigating to the package and then running `ls`. Now that your current directory is racing, you may want to have a look into the contents of the flatland packages.
```bash
rosls flatland<<< HIT TAB KEY TWICE TO SHOW OPTIONS >>>
```
You see the available packages for which you could use `rosls`. Complete the command with the `flatland_msgs` package. You should get the following output:
```bash
$ rosls flatland_msgs/
CMakeLists.txt  msg  package.xml  srv
```
You see:
*  `package.xml`: contains meta-information about the package, like dependencies to other packages, version, licence, etc.
*  `CMakeLists.txt`: contains information about how this package should be built.
*  `msg` and `srv` directories, which contain message and service files. These are covered later.


## ROS Master
The [ROS master](https://wiki.ros.org/Master) is responsible for letting individual ROS nodes find one another, and much more. It needs to run before any other node can be executed. You can start it by calling:
```bash
roscore
```
Once ROS master has helped individual nodes to locate one another, they can directly communicate with each other.
Furthermore, the `roscore` command starts 
* ROS Core Services (not important here)
* Parameter Server (covered later)
* Logging Node (covered later)

:information_source: Why could you start ROS nodes at the end of your installation process without calling `roscore`? --> `.launch`-files automatically start a new ROS master if there is none available.

:warning: The `roscore` command will block your current terminal window. You have to open a new terminal or a new terminal tab to launch a new node or ROS command.


## ROS Nodes
A [node](https://wiki.ros.org/Nodes) is a process that performs computation (quote from official tutorial).
The multiple nodes running in an automated vehicle are combined together into a graph/network and communicate with one another using messages, services, and the parameter server.
For example, our project consists of the nodes
*  flatland_server
*  vehicle_controller_node
*  (and more)

Nodes can be started and stopped independently of each other, which ensures stability if one node crashes.

Again, start the flatland simulation using `roslaunch` (, which is explained later).
```bash
roslaunch racing flatland_simulation.launch
```

Now, instead of starting a node with a launch file, we can also execute a node like the `vehicle_controller_node` using the `rosrun` command. Note that we have to specify the package in which the node is located.
```bash
# rosrun <package-name> <node-name>
rosrun racing vehicle_controller_node
```
:information_source: Why is there a warning, and why is the racing cart still not moving? --> This is because the `vehicle_controller_node` needs two specific *parameters* to work properly! We will set them in the next section.

:warning: The `rosrun` command blocks your current terminal window. You have to open a new terminal one to continue.


## ROS Parameter Server
Different nodes may need the same parameters to run properly. This is why ROS offers a *centralized/global* [parameter server](https://wiki.ros.org/Parameter%20Server). Check out how to interact with the parameter server by running:
```bash
rosparam
```
To make our `vehicle_controller_node` run properly, we first need to set the two parameters that it requires. 
```bash
rosparam set vehicle/sensor_topic vehicle/lidar_measurements
rosparam set vehicle/actuator_topic vehicle/actuator_commands
```
Now, you can see the parameters in the server's list:
```bash
rosparam list
```

## ROS Nodes Continued
Press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop the earlier unsuccessful `rosrun` command and then execute it again:
```bash
rosrun racing vehicle_controller_node
```

Now, the simulation node and the controller node interact with each other, as you can see in the visualization. Use the `rosnode` command to investigate nodes further:

```bash
rosnode
```
Print a list of all currently active nodes:
```bash
rosnode list
```
The result is:
```
/flatland_server            # 2D simulation
/rosout                     # handles logging of nodes, e.g. printing to terminal
/rviz                       # visualization
/vehicle_controller         # our node that got executed by rosrun above
/vehicle_state_publisher    # responsible for coordinate shift between map and vehicle
```
We look deeper into our controller node using the `rosnode info` command. Please execute:

```bash
rosnode info vehicle_controller
```
The output contains information about connectivity to other nodes, in particular *publications* and *subscriptions* of  *messages* through *topics*. These aspects are covered in later paragraphs.



## ROS Messages
Nodes communicate with each other by publishing [messages](https://wiki.ros.org/Messages) through topics.
Common messages are defined in [.msg](https://wiki.ros.org/msg)-files.
Standard primitive types (int32, int64, float32, float64, bool, ...) and arrays of them are supported.
Messages can also include other messages. 

Execute `rosmsg` to see the command's options.
```bash
rosmsg
```

Our `vehicle_controller_node` publishes a `geometry_msgs::Twist` message to the flatland simulation. This `Twist` message contains the vehicle's target velocity (in `linear.x`) and its target steering angle (in `angular.z`). To see this message definition, execute
```bash
rosmsg show geometry_msgs/Twist
```
The indentation represents a nested message, namely `geometry_msgs/Vector3`. You can check also its definition with
```bash
rosmsg show geometry_msgs/Vector3
```

Generally, messages can be defined:
*  by the official ROS distribution, like e.g. in the package `geometry_msgs`
*  or within a user package, like e.g. in the file `flatland/flatland_msgs/msg/Collision.msg`


## ROS Topics
[Topics](https://wiki.ros.org/Topics) are the communication channels through which nodes exchange messages. 
In general, nodes are not aware of which other node they are communicating with because topics are anonymous communication channels.

Quote from official tutorial: 
"Nodes that are interested in data *subscribe* to the relevant topic; nodes that generate data *publish* to the relevant topic.
There can be multiple publishers and subscribers to a topic.
Topics are intended for unidirectional, streaming communication."

Make sure the flatland simulation and the vehicle controller are running, like described before.

Topics are investigated with
```bash
rostopic
```
We want to print all available topics that are publised right now:
```bash
rostopic list
```
The displayed list should show a number of different topics, among which are `/vehicle/actuator_commands`
and `/vehicle/lidar_measurements`, which we have stored earlier as parameters on the parameter server.

For information about the `.msg` type that is sent through a topic, and about the topic's publishers and subscribers, run e.g.
```bash
rostopic info /vehicle/actuator_commands
```
You should see that the `vehicle_controller` publishes the message type `geometry_msgs/Twist` to the topic, and the `flatland_server` subscribes to it.
You can print the actual transmitted data on that topic to the terminal with

```bash
rostopic echo /vehicle/actuator_commands
```

:warning: The `rostopic echo` command will block your current terminal window. Hit <kbd>Ctrl</kbd> + <kbd>C</kbd> to cancel.


Instead of letting the `vehicle_controller_node` publish the actuator commands, we will now do that manually.
Terminate the previously executed `vehicle_controller` by hitting <kbd>Ctrl</kbd> + <kbd>C</kbd> in the terminal in which the `vehicle_controller_node` is still running.
The cart should now continue driving with the latest actuator commands until it crashes into a wall.
The topic `/vehicle/actuator_commands` is now free for you to publish:

```bash
rostopic pub /vehicle/actuator_commands <<< HIT TAB FOR AUTO-COMPLETE OF THE MESSAGE TEMPLATE >>>
```
Try to steer some turns on your own. Leave the message format as it has popped up and play around with the values. To navigate through the lines of the message, use the <kbd>Left</kbd>  and <kbd>Right</kbd> keys, otherwise the terminal thinks you want to execute another command.
The actually used fields of the message are:
* `linear.x`: target speed of the cart [m/s]
* `angular.z`: target steering angle of the cart [rad, positive counterclockwise]

While navigating the cart manually through the track, you may feel the need to reset it to its initial position. Luckily, the `flatland_server` offers this in form of a ROS service. 

## ROS Services

[Services](https://wiki.ros.org/Services) are similar to the publish/subscribe message system, but are intended for one-time execution instead of permanent streaming. A client directly calls a service at another node without going through an anonymous topic. By requesting a service, you are explicitly asking another process to perform a computation for you. Services also natively include responses, which is not the case for the publish/subscribe system.  

First, check out the syntax of
```bash
rosservice 
```

We want to find the corresponding service for resetting the cart's position in the list of all available services:
```bash
rosservice list
```
The service `/move_model` could be what we need. For more information about it:
```bash
rosservice info /move_model
```

The output confirms our guess further. Similar to `.msg` types, services work with `.srv` types. The type of the `/move_model` service is `flatland_msgs/MoveModel`. We can investigate this service type with the `rossrv` command (different from `rosservice`!)
```bash
rossrv info flatland_msgs/MoveModel
```
The input types of the service are shown above the separator `---`, and the return types are below it.

Call the service now like this:
```bash
rosservice call /move_model <<< HIT TAB FOR AUTO-COMPLETE OF THE SERVICE INPUT TEMPLATE >>>
```
Feed the service by manually inserting the following input data:
```
name: 'racing_cart'
pose:
  x: 2.0
  y: 6.0
  theta: -1.57
```
Hit <kbd>Enter</kbd> to execute the service and observe the return data in the terminal. You can use the :arrow_up_small:  key for getting back this service command to modify and play a bit with it.

## ROS Launch

"[roslaunch](https://wiki.ros.org/roslaunch) is a tool for easily launching multiple ROS nodes [...], as well as setting parameters on the Parameter Server. [...] `roslaunch` takes in one or more XML configuration files (with the `.launch` extension) that specify the parameters to set and nodes to launch [...]." (quote from official tutorial)

We have used `roslaunch` already when executing
```bash
roslaunch racing flatland_simulation.launch
```

Have a look into the argument XML file, `flatland_simulation.launch`, by executing `roscat`:
```bash
roscat racing flatland_simulation.launch
``` 

The most important [XML tags](https://wiki.ros.org/roslaunch/XML) of a `.launch` file are:
*  **node** for running nodes
*  **param** for setting parameters to the Parameter Server
*  **arg** for taking in arguments from outside this file, and otherwise setting default 

### Tag: node
We analyze the arguments of the **node** tag in this example line from our .launch file
```xml
<node name="vehicle_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>
```
*  `name` sets the display name of this node, like shown in `rosnode list`
*  `pkg` is the package name in which the node is located
*  `type` defines the actual node to be started

This example executes the node `robot_state_publisher` from the package `robot_state_publisher` and calls this instance `vehicle_state_publisher`.

### Tag: param

The **param** tag is often used *before* or *inside* the node tag, as it sets parameters that are typically used by the related node. 

```xml
<param name="use_sim_time" value="true"/>  
```
For example, the ROS-native parameter `use_sim_time` is set `true` here, such that all nodes follow a simulated time provided in a `/clock` topic by the flatland simulation. Note: if `use_sim_time` is `false`, all nodes would use the machine's system time.  


### Tag: arg

The `arg` tag enables default parameter values, as well as overwriting these with individual parameter choices given outside the .launch file.

```xml
<arg name="world_path"   default="$(find racing)/resources/racing_world.yaml"/>
```
The default value for the *argument* `world_path` is the path to the file `resources/racing_world.yaml`, which located in the `racing` package. Later on in the .launch file, this argument is put into the `world_path` *parameter* of the same name. 
```xml
<param name="world_path" value="$(arg world_path)" />
```
Note that this param tag lies *inside* the node tag "flatland_server", which makes the final name of the parameter on the Parameter Server `flatland_server/world_path`.

Now, launch the flatland simulation from the command line with different arguments:
```bash
roslaunch racing flatland_simulation.launch <<< HIT TAB TWICE TO SEE THE ARGUMENT LIST >>>
```
Complete the command with an invalid argument to see the effect:
```bash
roslaunch racing flatland_simulation.launch world_path:="dummy.txt"
```

## Task 1
So far, your task was just reading and trying out the commands that we proposed above. Now we challenge you to start with your first programming task. With the current commands from this lecture we started the flatland simulation and the vehicle controller with two separate launch files,

* `flatland_simulation.launch`
* `racing_controller.launch`

which needes the manual handling of two different terminals.

Your task is, to create a new combined launch file 

* `~ws/catkin_workspace/src/workshops/section_1/racing/launch/combined.launch` 

which starts the flatland simulation and the racing controller together. 


Notes:
- Use [`include`](https://wiki.ros.org/roslaunch/XML/include) tags instead of copying the entire contents!
- All arguments of the included `.launch` files should be adjustable from the calling `combined.launch` file.


Then execute and test you new launchfile by calling 
```bash
roslaunch racing combined.launch
```

## Wrap-up
- [ ] You learned that we can use `rospack find` and `roscd` to navigate through a ROS workspace
- [ ] You learned that the ROS Master can be evoked with `roscore`
- [ ] You learned that we can start ROS Nodes with `rosrun <package-name> <node-name>`
- [ ] You learned that ROS Messages contain information that are send over ROS Topics
- [ ] You learned that launch files are used to combine several ROS commands such as `rosrun` and `rosparam` into one executable file
- [ ] You learned that launch files are launched with `roslaunch`