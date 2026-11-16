# Robot Arm Kinematics and Pick-and-Place Simulator

An educational Python simulation of a 2D planar, two-link (2R) robotic manipulator performing conveyor-belt pick-and-place tasks.

## Planned capabilities

- Forward kinematics (joint angles -> end-effector coordinates)
- Inverse kinematics (target coordinates -> joint-angle solutions)
- 2D homogeneous transformations
- Workspace and joint-limit reachability checks
- Point-to-point (joint-space) and continuous Cartesian trajectories
- Interactive two-finger gripper and pick-and-place scene
- Motion metrics and simulation-data export

## Robot model

The initial model is a planar 2R manipulator. Link lengths and joint limits are educational parameters stored in `data/robot_config.json`; they are not specifications for a physical industrial robot.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Project status

The repository is being developed incrementally. The first milestone is the tested mathematical kinematics engine. The Streamlit dashboard and animated pick-and-place scene will be added after the core calculations are validated.

## Scope and limitations

This is a 2D educational simulator. It does not control physical hardware and does not model full 3D dynamics, forces, or industrial safety systems.

## Dataset

The simulator will generate its own reproducible CSV data from the configured robot model and simulated trajectories. These records are simulation outputs, not real-world sensor measurements.
