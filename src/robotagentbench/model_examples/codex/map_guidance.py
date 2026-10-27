GUIDANCE='''
READ-ONLY EXPLORED MAP (no navigation action assistance):
Every observation includes a top-down map: white known free floor, dark known
static obstacles, grey unknown. Red circle/line = your measured position and
heading. Blue line = your actual past trajectory, NOT a planned/recommended route.
Grid lines are world metres. This map is geometry-derived and revealed by the
measured head camera frustum/occlusion; it is NOT RGB SLAM. Full map is OFF.
Generated movable objects, task labels, task list and target poses are excluded.
You must choose your own route using the map AND cameras and execute it using
the ORIGINAL move_base/turn_base or raw robot actions. There is NO goto,
approach, path planner, automatic escape or geometry-derived pixel depth tool
available to you. No tool tucks arms or chooses a grasp for you.
The map can be stale around movable objects and does not guarantee full arm
clearance. Preserve safety checks and inspect real images. Physics, scores,
20 Hz controls and the 120-second native no-score-progress timer are unchanged;
all your executed navigation still consumes that simulation-time budget.
'''
