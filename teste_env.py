import pybullet as p
import pybullet_data
import time
import multiprocessing
import os  # For process ID
import random

# Define a simple model path (e.g., a sphere or a standard URDF)
# Ensure PyBullet can find this. Using pybullet_data is reliable.
# For this example, let's just use a basic plane and create a sphere.
# If you have a simple URDF for testing (like 'r2d2.urdf' from pybullet_data), use that.
# For simplicity, we'll just create a sphere programmatically.


def simulate_single_instance(task_id: int, simulation_steps: int = 100) -> tuple:
    """
    This function is executed by each worker process.
    It runs an independent PyBullet simulation.
    """
    # Each process gets its own physics client
    # p.DIRECT means no GUI, runs in the background
    physics_client_id = p.connect(p.DIRECT)
    if physics_client_id < 0:
        print(f"[Process {os.getpid()}] Error: Could not connect to PyBullet for task {task_id}.")
        return task_id, None

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load a plane
    plane_id = p.loadURDF("plane.urdf", physicsClientId=physics_client_id)

    # Create a simple sphere
    start_pos = [random.uniform(-1, 1), random.uniform(-1, 1), 2]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    sphere_radius = 0.1
    sphere_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                                 radius=sphere_radius,
                                                 rgbaColor=[1, 0, 0, 1],
                                                 physicsClientId=physics_client_id)
    sphere_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE,
                                                       radius=sphere_radius,
                                                       physicsClientId=physics_client_id)

    sphere_body_id = p.createMultiBody(baseMass=1,
                                       baseCollisionShapeIndex=sphere_collision_shape_id,
                                       baseVisualShapeIndex=sphere_visual_shape_id,
                                       basePosition=start_pos,
                                       baseOrientation=start_orientation,
                                       physicsClientId=physics_client_id)

    # Run simulation for a few steps
    final_position = None
    for _ in range(simulation_steps):
        p.stepSimulation(physicsClientId=physics_client_id)
        # In a real scenario, you might apply actions or get sensor data here

    # Get the final state (e.g., position of the sphere)
    if sphere_body_id is not None:  # Check if body was created
        try:
            final_position, _ = p.getBasePositionAndOrientation(sphere_body_id, physicsClientId=physics_client_id)
        except p.error as e:
            print(f"[Process {os.getpid()}] PyBullet error getting position for task {task_id}: {e}")
            final_position = None

    # Disconnect from this PyBullet instance
    p.disconnect(physicsClientId=physics_client_id)

    print(f"[Process {os.getpid()}] Task {task_id} finished. Final position: {final_position}")
    return task_id, final_position


if __name__ == "__main__":
    # --- Configuration ---
    num_tasks = 10  # Number of independent simulations to run (e.g., your n_individuals)
    # Number of processes to use. os.cpu_count() is a good starting point.
    # Be mindful not to create too many if simulations are heavy.
    num_processes = min(num_tasks, os.cpu_count() if os.cpu_count() is not None else 4)

    print(f"Main Process ID: {os.getpid()}")
    print(f"Starting {num_tasks} tasks using {num_processes} worker processes.\n")

    # Create a list of arguments for each task.
    # Here, each task is just identified by an ID and has the same number of sim steps.
    # In your case, this could be a list of rule_params_list for your CEM.
    tasks_args = [(i, 200 + i * 10) for i in range(num_tasks)]  # (task_id, num_simulation_steps)

    start_time = time.time()

    # --- Using multiprocessing.Pool ---
    # The 'spawn' start method is often more stable on different platforms, especially with PyBullet.
    # You might need to set this globally before any multiprocessing code runs.
    # try:
    #     multiprocessing.set_start_method('spawn', force=True)
    # except RuntimeError:
    #     print("Note: Could not force 'spawn' start method (might already be set or not applicable).")
    #     pass

    all_results = []
    # Create a pool of worker processes
    # The 'with' statement ensures the pool is properly closed
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Distribute the tasks to the worker processes
        # pool.starmap is useful when your worker function takes multiple arguments
        # It unpacks the tuples from tasks_args for each call to simulate_single_instance
        try:
            # starmap blocks until all results are collected
            results_from_pool = pool.starmap(simulate_single_instance, tasks_args)
            all_results.extend(results_from_pool)
        except Exception as e:
            print(f"An error occurred during multiprocessing: {e}")
            pool.terminate()  # Terminate the pool in case of an error
            pool.join()  # Wait for processes to terminate

    end_time = time.time()

    print(f"\n--- All tasks completed ---")
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    print("\nResults:")
    for task_id, final_pos in all_results:
        if final_pos:
            print(f"  Task {task_id}: Final Position = ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
        else:
            print(f"  Task {task_id}: Failed or no position data.")
