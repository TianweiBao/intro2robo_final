"""MuJoCo simulation of a Kinova Gen3 lite moving colored liquid with a clamped manual micropipette.

Modules:
    specs           geometry and constants (pipette, tip, tubes, dish, liquid colors)
    scene_builder   generates models/scene.xml
    sim             PipetteSim: liquid bookkeeping, contacts, rendering, task state
    liquid          container volumes and liquid visuals
    ik              inverse kinematics with the pipette held vertical
    motion          straight-line and arc motion primitives (IK + position actuators)
    scripted        the three-phase task: aspirate, transfer, dispense
    randomization   domain randomization for training and sim-to-real
"""
