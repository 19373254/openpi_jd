import collections
import dataclasses
import logging
import math
import pathlib

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("ok")
    # element = {
    #             "observation.images.cam_high": np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8),
    #             "observation.images.cam_left_wrist": np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8),
    #             "observation.images.cam_right_wrist": np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8),
    #             "observation.state": np.random.uniform(0, 1, 14),
    #             "prompt": 'debug',
    #         }
    element =  {
                "images": {
                    "cam_high": np.random.randint(0, 256, (3,300,300), dtype=np.uint8),
                    "cam_left_wrist":np.random.randint(0, 256, (3,300,300), dtype=np.uint8),
                    "cam_right_wrist": np.random.randint(0, 256, (3,300,300), dtype=np.uint8),
                },  
                "state": np.random.uniform(0, 1, 14),
                "prompt":'debug'
                }

    # Query model to get action
    action_chunk = client.infer(element)["actions"]
    print(action_chunk)

#         for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
#             logging.info(f"\nTask: {task_description}")

#             # Reset environment
#             env.reset()
#             action_plan = collections.deque()

#             # Set initial states
#             obs = env.set_init_state(initial_states[episode_idx])

#             # Setup
#             t = 0
#             replay_images = []

#             logging.info(f"Starting episode {task_episodes+1}...")
#             while t < max_steps + args.num_steps_wait:

#                     # Get preprocessed image
#                     # IMPORTANT: rotate 180 degrees to match train preprocessing
#                     img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
#                     wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
#                     img = image_tools.convert_to_uint8(
#                         image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
#                     )
#                     wrist_img = image_tools.convert_to_uint8(
#                         image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
#                     )

#                     # Save preprocessed image for replay video
#                     replay_images.append(img)

#                     if not action_plan:
#                         element = {
#                             "observation.images.cam_high": np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8),
#                             "observation.images.cam_left_wrist": np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8),
#                             "observation.images.cam_right_wrist": np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8),
#                             "observation.state": np.random.uniform(0, 1, 14),
#                             "prompt": 'debug',
#                         }

#                         # Query model to get action
#                         action_chunk = client.infer(element)["actions"]
#                         assert (
#                             len(action_chunk) >= args.replan_steps
#                         ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
#                         action_plan.extend(action_chunk[: args.replan_steps])

#                     action = action_plan.popleft()

#                     # Execute action in environment
#                     obs, reward, done, info = env.step(action.tolist())
#                     if done:
#                         task_successes += 1
#                         total_successes += 1
#                         break
#                     t += 1

#                 except Exception as e:
#                     logging.error(f"Caught exception: {e}")
#                     break

#             task_episodes += 1
#             total_episodes += 1

#             # Save a replay video of the episode
#             suffix = "success" if done else "failure"
#             task_segment = task_description.replace(" ", "_")
#             imageio.mimwrite(
#                 pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
#                 [np.asarray(x) for x in replay_images],
#                 fps=10,
#             )

#             # Log current results
#             logging.info(f"Success: {done}")
#             logging.info(f"# episodes completed so far: {total_episodes}")
#             logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

#         # Log final results
#         logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
#         logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

#     logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
#     logging.info(f"Total episodes: {total_episodes}")


# def _get_libero_env(task, resolution, seed):
#     """Initializes and returns the LIBERO environment, along with the task description."""
#     task_description = task.language
#     task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
#     env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
#     env = OffScreenRenderEnv(**env_args)
#     env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
#     return env, task_description


# def _quat2axisangle(quat):
#     """
#     Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
#     """
#     # clip quaternion
#     if quat[3] > 1.0:
#         quat[3] = 1.0
#     elif quat[3] < -1.0:
#         quat[3] = -1.0

#     den = np.sqrt(1.0 - quat[3] * quat[3])
#     if math.isclose(den, 0.0):
#         # This is (close to) a zero degree rotation, immediately return
#         return np.zeros(3)

#     return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)



