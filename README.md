# SCARA Robot Arm Assembly-Line Simulator

**Subject:** Robot Kinematics and Control (Mini Project 2)  
**Stack:** Python, Streamlit, Plotly, NumPy, pandas  
**Theme:** Light "morning in the factory" aesthetic (Cream `#fff8ea`, soft grey conveyors, blue Link 1, orange Link 2, cardboard-brown boxes)

---

## 🏭 1. Project Overview

This simulator provides an interactive, mathematically rigorous simulation of a **planar 2R SCARA-style robotic arm** with a parallel two-jaw gripper working on an industrial assembly line:
- **Scene:** 2D top-view workcell with an input conveyor belt on the left, an output conveyor belt on the right, and the robot base centered at $(0, 0)$.
- **Conveyor Workflow:** Cardboard-brown boxes arrive on the input belt, queue up, and stop at the marked **PICK station**. The robot arm picks each box, transfers it to the marked **PLACE station** on the output belt, and releases it as it slides downstream.
- **Dual Operation Modes:**
  - **Manual Mode:** Forward kinematics control via joint sliders/inputs with live gripper distance check, visual pick-and-place tracking, and hint calculation.
  - **Automatic Mode:** Inverse kinematics and quintic polynomial trajectory execution across Point-to-Point (joint arc), Continuous Path (Cartesian straight line), or both consecutively.

---

## 🚀 2. Setup and Execution

### Local Environment Setup
```bash
# Navigate to the project root directory
cd robot-arm-kinematics-simulator

# Create and activate a Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# Install required dependencies
pip install -r requirements.txt
```

### Running the Simulator
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 📦 3. Features & User Interface

The application is structured into 3 dedicated tabs:

### Tab 1: Assembly Line (The Simulation)
- **Mode Selector:** Choose between **Manual Mode** and **Automatic Mode**.
- **Manual Mode:**
  - Inputs for Joint 1 ($\theta_1$) and Joint 2 ($\theta_2$) angles in degrees.
  - `Move arm` and `Reset` buttons.
  - **Proximity Grasping:** If the gripper reaches within $0.05\text{ m}$ of the box at PICK, the jaws close and grasp the box. Moving to within $0.05\text{ m}$ of PLACE releases the box onto the output conveyor and increments the delivery counter.
  - **Real-Time Distance Readout:** Shows exact distance from the active target.
  - **Show Hint:** Solves inverse kinematics and displays the exact target angles.
- **Automatic Mode:**
  - Automatic inverse kinematics solver for pick and place targets.
  - **Prominent Trajectory Selector:**
    1. *Point-to-point (joint-space interpolation, curved tool path)*
    2. *Continuous path (straight line, IK at every waypoint)*
    3. *Both (consecutive cycles of each)*
  - **Dynamic Path Highlighting:** The chosen trajectory is drawn thicker and clearly traced by the robot arm during task execution.
  - **Playback Controls:** `▶ Play`, `⏸ Pause`, `↺ Reset`, and an interactive timeline scrubbing slider.
- **Real-Time Status Banner:** Live display of the current step description, $\theta_1$, $\theta_2$, gripper $(X, Y)$ position, and gripper state (`Open` / `Closed (Holding Box)`).
- **2D Top-View Factory Floor Plan:** Authentic industrial floor plan rendering with dark rubber belts, side rails, rollers, mounting legs, and cleanly positioned labels (preventing any overlap with boxes). Axes, ticks, and gridlines are hidden for a clean factory floor appearance, with a clear horizontal legend placed under the picture.
- **"How to read the picture":** Informative caption under the canvas for fast student viva and demonstration understanding.

### Tab 2: Kinematics and Trajectory Analysis
- **Forward Kinematics (FK):** Angle sliders with live $(X, Y)$ position, radial reach, orientation $\phi$, and live $3 \times 3$ planar homogeneous transformation matrices:
  - $T_{01} = R_z(\theta_1) \cdot T_x(L_1)$
  - $T_{12} = R_z(\theta_2) \cdot T_x(L_2)$
  - $T_{02} = T_{01} \cdot T_{12}$
- **Inverse Kinematics (IK) & Reachability:** Interactive target coordinate $(X, Y)$ inputs with reachability validation ($|L_1 - L_2| \le r \le L_1 + L_2$), $c_2 = \cos(\theta_2)$, and both **Elbow-Right** ($\theta_2 \ge 0$) and **Elbow-Left** ($\theta_2 \le 0$) angle solutions.
- **PTP vs. Continuous Benchmark:** Complete quantitative comparison table covering path length, straight-line distance, path efficiency $\%$, maximum deviation from the line, joint travel, peak joint speeds ($\text{deg/s}$), peak tool speed ($\text{m/s}$), feasibility, and an engineering verdict.
- **Joint Angles vs. Time Profile:** Plotly curves showing $\theta_1(t)$ and $\theta_2(t)$ over time.

### Tab 3: Documentation
- Comprehensive engineering documentation with SCARA parameters, assumptions, Denavit-Hartenberg (DH) table, analytical FK/IK derivations, transformation explanations, quintic polynomial time scaling, and performance metrics definitions.

---

## 🧮 4. Mathematical Equations

### Forward Kinematics (FK)
$$x = L_1 \cos(\theta_1) + L_2 \cos(\theta_1 + \theta_2)$$
$$y = L_1 \sin(\theta_1) + L_2 \sin(\theta_1 + \theta_2)$$
$$\phi = \theta_1 + \theta_2$$

### Inverse Kinematics (IK)
$$c_2 = \frac{x^2 + y^2 - L_1^2 - L_2^2}{2 L_1 L_2}$$
$$\theta_2 = \pm \arccos(c_2)$$
$$\theta_1 = \text{atan2}(y, x) - \text{atan2}(L_2 \sin\theta_2, L_1 + L_2 \cos\theta_2)$$

### Quintic Time Scaling
$$s(u) = 10u^3 - 15u^4 + 6u^5, \quad u \in [0, 1]$$
$$\dot{s}(u) = 30u^2 - 60u^3 + 30u^4$$
$$\ddot{s}(u) = 60u - 180u^2 + 120u^3$$
Guarantees zero velocity and zero acceleration at movement boundaries.

---

## 🐙 5. GitHub Publishing Instructions

To publish this project to GitHub, execute the following commands in your terminal:

```bash
# 1. Initialize local Git repository
git init

# 2. Stage all project files
git add .

# 3. Commit the implementation
git commit -m "Complete SCARA Robot Arm Assembly-Line Simulator"

# 4. Rename default branch to main
git branch -M main

# 5. Connect to your GitHub repository (replace with your repository URL)
git remote add origin <repo-url>

# 6. Push the code to GitHub
git push -u origin main
```
Screenshot of the output
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c9e5e8aa-5a86-4e18-9351-df71e7a79f2a" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/002370b9-d218-4abc-8461-958c67292bd1" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cd638773-d652-40c1-bd3b-6b4bc22c94df" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/00b520ce-5621-47c7-a98f-37b501690bb7" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/669179ef-d4a1-402b-9938-098f004d90c5" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8b73712f-07af-433c-a855-1e7b23717003" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/bd4e1264-b173-47d7-b619-c951e1f490cb" />







