---
title: "ROS, ROS2, and the Art of Wasting Time"
description: "A scathing critique of ACDC's decision to introduce both ROS and ROS2 in Week 1—because who doesn’t love learning outdated tech alongside its replacement?"
slug: acdc-edX-ros
authors: [CagriCatik]
tags: [Autonomous Driving, edX, RWTH Aachen]
---

ROS, ROS2, and the Fine Line Between Ambition and Absurdity. 


<!-- truncate -->

Week 1 of ACDC introduces you to the *foundation* of robotics development: the Robot Operating System (ROS). But why stop at one system when you can overload students with **two**? That’s right—ACDC has decided to teach both ROS and ROS2 at the same time, in what can only be described as an exercise in futility.

## ROS vs. ROS2: A Tale of Two Systems

For the uninitiated, ROS (Robot Operating System) is an open-source framework for robotics development that’s been around for over a decade. ROS2 is its younger, shinier sibling, designed to address the flaws of its predecessor. Sounds great, right? Except there’s one glaring problem:

**ROS is on its way out.** Its most recent distribution, Noetic, reaches end-of-life in 2025. Meanwhile, ROS2 is actively being developed, with support for modern features like real-time communication and multi-threading.

So why, in the name of all that is logical, does ACDC insist on teaching both simultaneously? Are we supposed to master outdated tech while also grappling with its replacement? If ROS2 is the future, why not just focus on that?

## The Problem with Teaching Two Systems at Once

1. **Split Focus, Split Sanity**  
   Learning one complex framework is hard enough. Introducing two frameworks with overlapping but subtly different concepts is a recipe for confusion. Are we supposed to focus on ROS's `roslaunch` or ROS2's `ros2 launch`? Should we learn about `roscore` even though ROS2 doesn’t use it? ACDC offers no clear guidance here—just a giant heap of terms and tools.

2. **Outdated Knowledge**  
   ROS is like a flip phone in the age of smartphones—technically functional, but largely irrelevant. By the time students complete this course, their ROS knowledge will be as useful as knowing how to set up a MySpace page. Why waste time on a system that’s already fading into obscurity?.

3. **ROS2 is Superior (Mostly)**  
   ROS2 isn’t just a minor update; it’s a ground-up redesign. It offers critical improvements like real-time capabilities, better security, and cross-platform support. Teaching ROS alongside ROS2 feels like teaching someone how to drive stick shift right before they buy an electric car. It’s not just unnecessary—it’s counterproductive.

4. **The "Both Are Useful" Fallacy**  
   The course tries to justify this madness by suggesting that both systems are still relevant. Sure, some legacy projects still use ROS, but industry trends clearly favor ROS2. If the goal is to prepare students for the future, why anchor them to the past?

## The Practical Nightmare

Let’s talk about implementation. Week 1 expects you to dive into ROS basics like nodes, topics, and messages, while also introducing ROS2 features like DDS (Data Distribution Service). The result? Pure chaos.

- **Tool Overload**: You’re forced to install and juggle tools for both systems, each with their quirks. Want to use RVIZ? Make sure you’re running the right version for the right system.
- **Conflicting Commands**: ROS commands start with `ros`, while ROS2 commands start with `ros2`. Simple enough—until you accidentally mix them up and spend an hour troubleshooting why your topic isn’t publishing.
- **Double the Bugs**: ROS is infamous for its dependency hell. ROS2 improves on this but introduces its own set of quirks. Now imagine dealing with bugs from both systems simultaneously. Fun times.

## The Missed Opportunity

This could’ve been a fantastic chance to focus entirely on ROS2, equipping students with cutting-edge skills. Instead, the course wastes time teaching a dying system, leaving students with a fragmented understanding of both. It’s like being handed two half-built bicycles and told to ride them at the same time.

## The Objective Perspective (Because Fairness is Overrated)

Yes, ROS has historical significance, and some industries still use it. But the decision to teach both ROS and ROS2 in parallel lacks foresight. ACDC claims to prepare students for the future, but clinging to ROS feels like preparing for a horse-drawn carriage comeback.

## Final Thoughts: Pick a Lane, ACDC

The simultaneous introduction of ROS and ROS2 isn’t just unnecessary—it’s a disservice to students. It splits focus, wastes time, and muddles the learning experience. If ACDC wants to be taken seriously, it needs to stop living in the past and commit to the future.

For now, we’ll keep flipping between ROS and ROS2, wondering if this dual approach is an elaborate prank. Thanks, ACDC.

---

> *Disclaimer: This review contains no exaggerations—just cold, hard truths.*
