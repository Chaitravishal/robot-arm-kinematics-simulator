# Robot Arm Kinematics and Trajectory Planning Simulator

## About the Project

This project is a Python-based robot arm simulator designed to demonstrate how robotic arms work in industrial environments.

The simulator shows how a robotic arm moves objects from one location to another using a conveyor belt system. Users can control the robot, choose object pickup and drop locations, and watch the complete process.

It also allows users to enter robot joint angles or target positions and see how the robot responds.

## Project Objectives

- Understand how robotic arms work.
- Calculate the position of the robot's gripper.
- Find the required joint angles for a given position.
- Check whether a location can be reached by the robot.
- Plan and display robot movements.
- Simulate object pickup and placement.
- Compare different ways of moving the robot.

## Main Features

- **Interactive Robot Simulation:** View and control a robotic arm in a factory environment.
- **Joint Angle Control:** Change the robot's joint angles and observe its movement.
- **Position Control:** Enter a target location and let the simulator calculate how the robot should move.
- **Reachability Check:** Identify whether the robot can reach a selected location.
- **Movement Planning:** Display different ways for the robot to move between locations.
- **Pick and Place:** Simulate picking up an object from a conveyor belt and placing it at another location.
- **Performance Comparison:** Compare robot movements based on time, distance, and smoothness.
- **Visual Dashboard:** View the robot, conveyor belt, objects, and simulation information in one place.

## How the Simulation Works

1. The user starts the simulation.
2. An object arrives at the pickup location on the conveyor belt.
3. The robot calculates how to reach the object.
4. The robotic arm moves towards the object.
5. The gripper closes and picks up the object.
6. The robot carries the object to the selected destination.
7. The gripper opens and releases the object.
8. The simulation displays the completed operation.

Users can also control the robot manually by entering joint angles or target positions.

## Tools and Technologies

- Python
- Streamlit
- NumPy
- Plotly
- Matplotlib
- Pandas

## How to Run the Project

1. Clone or download this repository.
2. Install the required Python packages.
3. Open the project folder.
4. Run the application using Streamlit.
5. Open the displayed local address in your browser.



