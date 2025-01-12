![ROS1](https://img.shields.io/badge/ROS1-blue)

In this task, we will continue with the **Object Association** step. Since we have already prepared everything necessary in the last task, we can start directly with the implementation.

## Task 2: Implement Object Association

Your task is to fill gaps in the C++ code of the [object association](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_3/object_fusion/src/modules/matcher/distance_measures/) module: `workshops/section_3/object_fusion/src/modules/matcher/distance_measures/`
There you will find the two distance measures
- [Intersection over Union](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_3/object_fusion/src/modules/matcher/distance_measures/IntersectionOverUnion.cpp#L40) `workshops/section_3/object_fusion/src/modules/matcher/distance_measures/IntersectionOverUnion.cpp`
- [Mahalanobis distance](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_3/object_fusion/src/modules/matcher/distance_measures/Mahalanobis.cpp#L69) `workshops/section_3/object_fusion/src/modules/matcher/distance_measures/Mahalanobis.cpp`

#### Association code

We already implemented large parts of the association code, however the calculations of both distance measures are still missing. Your turn!

1. Implement the IoU as described in the slides. By now you should be familiar enough with the code to manage this task without detailed instructions. To choose, which distance measure is used for the association, you have to edit the [fusion config](https://github.com/ika-rwth-aachen/acdc/blob/main/catkin_workspace/src/workshops/section_3/object_fusion_wrapper/param/fusion.yaml#L4). Here you can also find all other parameters used for the association.

2. Implement the Mahalanobis distance $d_{G,S}$ in the x-y-plane as described in the slides. Here you should also have no need for detailed instructions.

After completing each task, rebuild the workspace with `catkin build`. Both distance measures should now correcly assign the objects. However, without a subsequent fusion, the objects are still drifting. So let's continue with the next task to complete the object fusion!
