# SCARA Robot Arm Assembly-Line Simulator

## About the Project

The SCARA Robot Arm Assembly-Line Simulator is an interactive application that demonstrates how a robotic arm performs pick-and-place operations in an industrial assembly line.

The simulation consists of two conveyor belts and a robotic arm with a two-jaw gripper. Boxes arrive on the input conveyor, the robot picks them up, moves them to the output conveyor, and releases them.

Users can operate the robot manually or run automatic pick-and-place operations.

## Features

- **Assembly-Line Simulation:** Visualizes boxes moving between two conveyor belts.
- **Manual Mode:** Control the robotic arm by adjusting its joint angles.
- **Automatic Mode:** Run pick-and-place operations automatically.
- **Forward Kinematics:** Calculate the gripper position using joint angles.
- **Inverse Kinematics:** Calculate the joint angles required to reach a target position.
- **Trajectory Planning:** Compare point-to-point and continuous robot movements.
- **Simulation Controls:** Play, pause, reset, and control the simulation timeline.
- **Live Visualization:** Observe robot movement, gripper position, and trajectory paths.
- **Kinematics Analysis:** Explore robot positions, movement paths, and trajectory graphs.

## Screenshots

### 1. Main Assembly-Line Simulation

Shows the complete workcell, including the robotic arm, input conveyor, output conveyor, and boxes.

![Main Assembly-Line Simulation](https://github.com/user-attachments/assets/c9e5e8aa-5a86-4e18-9351-df71e7a79f2a)

### 2. Manual Robot Control

Allows users to adjust the robot's joint angles and control its movement.

![Manual Robot Control](https://github.com/user-attachments/assets/002370b9-d218-4abc-8461-958c67292bd1)

### 3. Gripper at the Pick Station

Shows the robotic gripper positioned to pick up a box from the input conveyor.

![Gripper at Pick Station](https://github.com/user-attachments/assets/72e31dfb-5f88-471c-9a0b-cca7aeae1dba)

### 4. Automatic Pick-and-Place

Demonstrates the robot automatically moving a box from the input conveyor to the output conveyor.

![Automatic Pick-and-Place](https://github.com/user-attachments/assets/cd638773-d652-40c1-bd3b-6b4bc22c94df)

### 5. Kinematics and Trajectory Analysis

Displays robot kinematics calculations and graphs used to understand its movement.

![Kinematics and Trajectory Analysis](https://github.com/user-attachments/assets/8b73712f-07af-433c-a855-1e7b23717003)

### 6. Project Documentation

Provides explanations of the robot model, kinematics, transformations, and trajectory planning.

![Project Documentation](https://github.com/user-attachments/assets/bd4e1264-b173-47d7-b619-c951e1f490cb)

## Technologies Used

- Python
- Streamlit
- Plotly
- NumPy
- pandas

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/Chaitravishal/robot-arm-kinematics-simulator.git
```

### Step 2: Open the Project Folder

```bash
cd robot-arm-kinematics-simulator
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Start the Application

```bash
streamlit run app.py
```

**SCARA Robot Arm Assembly-Line Simulator**  
An interactive demonstration of robotic movement, kinematics, and automated pick-and-place operations.
