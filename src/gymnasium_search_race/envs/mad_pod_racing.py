from itertools import product
from pathlib import Path
from typing import Any, SupportsFloat

from gymnasium import spaces
from gymnasium.core import ActType, ObsType
import numpy as np
import pygame

from gymnasium_search_race.envs.models import Car, Point, Unit
from gymnasium_search_race.envs.search_race import SCALE_FACTOR, SearchRaceEnv
from stable_baselines3 import PPO


ROOT_PATH = Path(__file__).resolve().parent
ASSETS_PATH = ROOT_PATH / "assets" / "mad_pod_racing"

BOOST_THRUST = 650
MAPS = [
    [
        [12460, 1350],
        [10540, 5980],
        [3580, 5180],
        [13580, 7600],
    ],
    [
        [3600, 5280],
        [13840, 5080],
        [10680, 2280],
        [8700, 7460],
        [7200, 2160],
    ],
    [
        [4560, 2180],
        [7350, 4940],
        [3320, 7230],
        [14580, 7700],
        [10560, 5060],
        [13100, 2320],
    ],
    [
        [5010, 5260],
        [11480, 6080],
        [9100, 1840],
    ],
    [
        [14660, 1410],
        [3450, 7220],
        [9420, 7240],
        [5970, 4240],
    ],
    [
        [3640, 4420],
        [8000, 7900],
        [13300, 5540],
        [9560, 1400],
    ],
    [
        [4100, 7420],
        [13500, 2340],
        [12940, 7220],
        [5640, 2580],
    ],
    [
        [14520, 7780],
        [6320, 4290],
        [7800, 860],
        [7660, 5970],
        [3140, 7540],
        [9520, 4380],
    ],
    [
        [10040, 5970],
        [13920, 1940],
        [8020, 3260],
        [2670, 7020],
    ],
    [
        [7500, 6940],
        [6000, 5360],
        [11300, 2820],
    ],
    [
        [4060, 4660],
        [13040, 1900],
        [6560, 7840],
        [7480, 1360],
        [12700, 7100],
    ],
    [
        [3020, 5190],
        [6280, 7760],
        [14100, 7760],
        [13880, 1220],
        [10240, 4920],
        [6100, 2200],
    ],
    [
        [10323, 3366],
        [11203, 5425],
        [7259, 6656],
        [5425, 2838],
    ],
]

START_POINT_MULT = [[500, -500], [-500, 500], [1500, -1500], [-1500, 1500]]


class MadPodRacingEnv(SearchRaceEnv):

  def __init__(
      self,
      render_mode: str | None = None,
      laps: int = 3,
      car_max_thrust: int = 200,
      test_id: int | None = None,
      sequential_maps: bool = False,
      opponent_path: str | Path | None = None,
      boost_on_first_move: bool = False,
      boost_opponent_on_first_move: bool = False,
      discover_checkpoints: bool = True,
      thrust_mapping: str = "linear",
      pruned_actions: bool = False,
      smart_thrust_cap: bool = False,
      align_reward_beta: float = 0.0,
      use_action_history: bool = False,
      smoothness_weight: float = 0.0,
      potential_alpha: float = 0.0,
      non_uniform_grid: int = 0,
      action_history_len: int = 0,
      use_asymmetric_history: bool = False,
      use_damped_thrust: bool = False,
      damped_thrust_beta: float = 0.9,
  ) -> None:
    super().__init__(
        render_mode=render_mode,
        laps=laps,
        car_max_thrust=car_max_thrust,
        test_id=test_id,
        sequential_maps=sequential_maps,
        discover_checkpoints=discover_checkpoints,
    )
    self.thrust_mapping = thrust_mapping
    self.pruned_actions = pruned_actions
    self.smart_thrust_cap = smart_thrust_cap
    self.align_reward_beta = align_reward_beta
    self.use_action_history = use_action_history
    self.smoothness_weight = smoothness_weight
    self.potential_alpha = potential_alpha
    self.non_uniform_grid = non_uniform_grid
    self.action_history_len = action_history_len
    self.use_asymmetric_history = use_asymmetric_history
    self.use_damped_thrust = use_damped_thrust
    self.damped_thrust_beta = damped_thrust_beta

    if self.use_action_history and self.action_history_len == 0:
      self.action_history_len = 1
    if self.use_asymmetric_history and self.action_history_len == 0:
      self.action_history_len = 1

    extra_dim = 0
    if self.use_asymmetric_history:
      extra_dim += self.action_history_len
      self.previous_actions = np.zeros(
          self.action_history_len, dtype=np.float64
      )
    elif self.action_history_len > 0:
      extra_dim += 2 * self.action_history_len
      self.previous_actions = np.zeros(
          2 * self.action_history_len, dtype=np.float64
      )

    if self.use_damped_thrust:
      extra_dim += 1
      self.damped_thrust = 0.0

    if extra_dim > 0:
      self.observation_space = spaces.Box(
          low=-1.0,
          high=1.0,
          shape=(10 + extra_dim,),
          dtype=np.float64,
      )
    self.car_radius = 400
    self.min_impulse = 120.0

    self.background_img_path = ASSETS_PATH / "background.jpg"
    self.car_img_path = ASSETS_PATH / "space_ship_runner.png"
    self.opponent_car_img_path = ASSETS_PATH / "space_ship_blocker.png"

    self.opponent_model = PPO.load(opponent_path) if opponent_path else None

    self.boost_on_first_move = boost_on_first_move
    self.boost_opponent_on_first_move = boost_opponent_on_first_move

  def _get_test_ids(self) -> list[int]:
    return list(range(len(MAPS)))

  def _get_test_checkpoints(self) -> list[np.ndarray]:
    return [
        np.array(checkpoints, dtype=self.observation_space.dtype)
        for checkpoints in MAPS
    ]

  def _get_runner_obs(self, car_index: int) -> ObsType:
    car = self.cars[car_index]
    obs = []

    # position and angle of the next checkpoints relative to the car
    next_cp_x, next_cp_y = self.checkpoints[
        (car.current_checkpoint + 1) % len(self.checkpoints)
    ]
    next_obs = self._get_diff_obs(car=car, x=next_cp_x, y=next_cp_y)
    obs.append(next_obs)
    if self.discover_checkpoints and car.current_checkpoint <= len(
        self.checkpoints
    ):
      # We are still in the first lap, we should not know the coordinates of the checkpoint after the next one yet.
      # Therefore, for our obs vector, we will just repeat the next checkpoint twice.
      obs.append(next_obs)
    else:
      second_next_cp_x, second_next_cp_y = self.checkpoints[
          (car.current_checkpoint + 2) % len(self.checkpoints)
      ]
      second_next_obs = self._get_diff_obs(
          car=car,
          x=second_next_cp_x,
          y=second_next_cp_y,
      )
      obs.append(second_next_obs)

    # car speed
    obs.append(self._get_speed_obs(car=car))

    if self.action_history_len > 0:
      obs.append(self.previous_actions)

    if self.use_damped_thrust:
      obs.append(
          np.array([self.damped_thrust], dtype=self.observation_space.dtype)
      )

    return np.concatenate(obs).astype(self.observation_space.dtype)

  def _dist_to_finish(self, cp_index: int, current_checkpoint: int) -> float:
    total = self.laps * len(self.checkpoints)
    remaining = total - current_checkpoint - 1
    dist = 0.0
    num_cps = len(self.checkpoints)
    for k in range(remaining):
      idx = (cp_index + k) % num_cps
      dist += self.cp_distances[idx]
    return dist

  def reset(
      self,
      *,
      seed: int | None = None,
      options: dict[str, Any] | None = None,
  ) -> tuple[ObsType, dict[str, Any]]:
    if self.use_asymmetric_history:
      self.previous_actions = np.zeros(
          self.action_history_len, dtype=np.float64
      )
    elif self.action_history_len > 0:
      self.previous_actions = np.zeros(
          2 * self.action_history_len, dtype=np.float64
      )

    if self.use_damped_thrust:
      self.damped_thrust = 0.0

    obs, info = super().reset(seed=seed, options=options)
    self.cp_distances = [
        np.linalg.norm(
            self.checkpoints[i]
            - self.checkpoints[(i + 1) % len(self.checkpoints)]
        )
        for i in range(len(self.checkpoints))
    ]
    if self.potential_alpha > 0.0:
      next_cp = self._get_next_checkpoint_index()
      cp = self.checkpoints[next_cp]
      dist = self.car.distance(Point(x=cp[0], y=cp[1]))
      self.current_potential = -self.potential_alpha * (
          dist + self._dist_to_finish(next_cp, 0)
      )
    return obs, info

  def step(
      self,
      action: ActType,
  ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    if self.potential_alpha > 0.0:
      prev_potential = self.current_potential

    if isinstance(action, (int, np.integer)):
      angle_phys, thrust_phys = self._convert_action_to_angle_thrust(action)
      cont_action = np.array(
          [
              angle_phys / self.max_rotation_per_turn,
              thrust_phys / self.car_max_thrust,
          ],
          dtype=np.float64,
      )
    else:
      cont_action = np.array(action, dtype=np.float64)

    if self.use_damped_thrust:
      if isinstance(action, (np.ndarray, list)) and len(action) >= 2:
        thrust_val = action[1]
      else:
        _, thrust_phys = self._convert_action_to_angle_thrust(action)
        thrust_val = thrust_phys / self.car_max_thrust
      self.damped_thrust = (
          self.damped_thrust_beta * self.damped_thrust
          + (1.0 - self.damped_thrust_beta) * thrust_val
      )

    if self.use_asymmetric_history:
      prev_action_copy = self.previous_actions[0]
      self.previous_actions = np.roll(self.previous_actions, 1)
      self.previous_actions[0] = cont_action[0]
    elif self.action_history_len > 0:
      prev_action_copy = self.previous_actions[:2].copy()
      self.previous_actions = np.roll(self.previous_actions, 2)
      self.previous_actions[:2] = cont_action

    obs, reward, terminated, truncated, info = super().step(action)

    if self.use_asymmetric_history and self.smoothness_weight > 0.0:
      smoothness_penalty = (
          -self.smoothness_weight * (cont_action[0] - prev_action_copy) ** 2
      )
      reward += smoothness_penalty
    elif self.action_history_len > 0 and self.smoothness_weight > 0.0:
      smoothness_penalty = (
          -self.smoothness_weight * (cont_action[0] - prev_action_copy[0]) ** 2
      )
      reward += smoothness_penalty

    if self.potential_alpha > 0.0:
      next_cp = self._get_next_checkpoint_index()
      dist = self.car.distance(
          Point(x=self.checkpoints[next_cp][0], y=self.checkpoints[next_cp][1])
      )
      self.current_potential = -self.potential_alpha * (
          dist + self._dist_to_finish(next_cp, self.car.current_checkpoint)
      )
      reward += self.current_potential - prev_potential

    return obs, reward, terminated, truncated, info

  def _get_blocker_obs(self, car_index: int) -> ObsType:
    runner_car_index = (car_index + 1) % len(self.cars)
    runner_car = self.cars[runner_car_index]
    blocker_car = self.cars[car_index]
    return np.concatenate([
        self._get_diff_obs(car=blocker_car, x=runner_car.x, y=runner_car.y),
        self._get_speed_obs(car=blocker_car),
        self._get_runner_obs(car_index=runner_car_index),
    ])

  def _get_obs(self) -> ObsType:
    return self._get_runner_obs(car_index=0)

  def _get_opponent_obs(self) -> ObsType:
    return self._get_blocker_obs(car_index=1)

  def _get_terminated(self) -> bool:
    return any(
        car.current_checkpoint >= self.total_checkpoints for car in self.cars
    )

  def _generate_checkpoints(
      self,
      options: dict[str, Any] | None = None,
  ) -> np.ndarray:
    # https://github.com/Agade09/CSB-Runner-Arena/blob/master/Arena.cpp#L276
    checkpoints = super()._generate_checkpoints(options=options)
    shift = self.np_random.integers(0, len(checkpoints))
    checkpoints = np.roll(checkpoints, shift=shift, axis=0)
    delta = self.np_random.integers(-30, 31, checkpoints.shape)
    return checkpoints + delta

  def _generate_car(self) -> None:
    # https://github.com/robostac/coders-strike-back-referee/blob/master/csbref.go#L407
    self.cars = []

    start_point_mult_index = self.np_random.choice(
        len(START_POINT_MULT),
        size=2,
        replace=False,
    )

    cp1_minus_cp0 = self.checkpoints[1] - self.checkpoints[0]
    distance = np.linalg.norm(cp1_minus_cp0)
    cp1_minus_cp0 /= distance

    car_start_point_mult = START_POINT_MULT[start_point_mult_index[0]]
    self.car = Car(
        x=self.checkpoints[0][0] + cp1_minus_cp0[1] * car_start_point_mult[0],
        y=self.checkpoints[0][1] + cp1_minus_cp0[0] * car_start_point_mult[1],
    )
    self.car.angle = self.car.get_angle(
        x=self.checkpoints[1][0],
        y=self.checkpoints[1][1],
    )
    self.cars.append(self.car)

    if self.opponent_model:
      opponent_start_point_mult = START_POINT_MULT[start_point_mult_index[1]]
      self.opponent_car = Car(
          x=self.checkpoints[0][0]
          + cp1_minus_cp0[1] * opponent_start_point_mult[0],
          y=self.checkpoints[0][1]
          + cp1_minus_cp0[0] * opponent_start_point_mult[1],
      )
      self.opponent_car.angle = self.opponent_car.get_angle(
          x=self.checkpoints[1][0],
          y=self.checkpoints[1][1],
      )
      self.cars.append(self.opponent_car)
    else:
      self.opponent_car = None

  def _adjust_car(self) -> None:
    for car in self.cars:
      car.round_position()
      car.truncate_speed(friction=self.car_friction)

  def _apply_angle_thrust(self, angle: float, thrust: float) -> None:
    if self.boost_on_first_move and self.episode_length == 0:
      thrust = BOOST_THRUST

    super()._apply_angle_thrust(angle=angle, thrust=thrust)

    if self.opponent_car:
      assert self.opponent_model is not None
      observation = self._get_opponent_obs()
      action, _ = self.opponent_model.predict(observation, deterministic=True)
      angle, thrust = self._convert_action_to_angle_thrust(action=action)

      if self.boost_opponent_on_first_move and self.episode_length == 0:
        thrust = BOOST_THRUST

      self.opponent_car.rotate(angle=angle)
      self.opponent_car.thrust_towards_heading(thrust=thrust)

  def _convert_action_to_angle_thrust(
      self,
      action: ActType,
  ) -> tuple[float, float]:
    angle, thrust_action = action
    assert -1.0 <= angle <= 1.0
    assert 0.0 <= thrust_action <= 1.0

    angle = np.rint(angle * self.max_rotation_per_turn)

    if self.thrust_mapping == "linear":
      thrust = thrust_action
    elif self.thrust_mapping == "power_0_2":
      thrust = np.power(thrust_action, 0.2)
    elif self.thrust_mapping == "power_0_5":
      thrust = np.power(thrust_action, 0.5)
    elif self.thrust_mapping == "cubic":
      thrust = np.power(thrust_action, 3.0)
    elif self.thrust_mapping == "threshold":
      thrust = 1.0 if thrust_action > 0.2 else 0.0
    else:
      raise ValueError(f"Unknown thrust mapping: {self.thrust_mapping}")

    thrust = np.rint(thrust * self.car_max_thrust)

    if self.smart_thrust_cap:
      next_cp_idx = (self.car.current_checkpoint + 1) % len(self.checkpoints)
      cp_x, cp_y = self.checkpoints[next_cp_idx]
      target_angle = self.car.get_angle(cp_x, cp_y)
      heading_error = target_angle - self.car.angle
      heading_error = (heading_error + 180) % 360 - 180
      if abs(heading_error) > 90:
        thrust = 0.0

    return angle, thrust

  def _get_collision_reward(self) -> SupportsFloat:
    return 0

  def _get_checkpoint_visit_reward(self, car_index: int) -> SupportsFloat:
    return 1 if car_index == 0 else 0

  def _move_car(self) -> SupportsFloat:
    target_cp_idx = self._get_next_checkpoint_index()

    if not self.opponent_car:
      reward = super()._move_car()
    else:
      reward = -0.1
      t = 0.0

      while t < 1.0:
        first_collision = None
        car_index = None

        car_collision = self.car.get_collision(
            self.opponent_car,
            radius=2 * self.car_radius,
        )
        if car_collision is not None and car_collision.time + t < 1.0:
          first_collision = car_collision

        for i, car in enumerate(self.cars):
          checkpoint_index = (car.current_checkpoint + 1) % len(
              self.checkpoints
          )
          next_checkpoint = Unit(
              x=self.checkpoints[checkpoint_index][0],
              y=self.checkpoints[checkpoint_index][1],
          )
          checkpoint_collision = car.get_collision(
              next_checkpoint,
              radius=self.checkpoint_radius,
          )

          if (
              checkpoint_collision is not None
              and checkpoint_collision.time + t < 1.0
              and (
                  first_collision is None
                  or checkpoint_collision.time < first_collision.time
              )
          ):
            first_collision = checkpoint_collision
            car_index = i

        if first_collision is None:
          for car in self.cars:
            car.move(t=1.0 - t)
          break

        for car in self.cars:
          car.move(t=first_collision.time)

        if isinstance(first_collision.second_unit, Car):
          first_collision.first_unit.bounce(
              first_collision.second_unit,
              min_impulse=self.min_impulse,
              min_radius=2 * self.car_radius,
          )
          reward += self._get_collision_reward()
        else:  # checkpoint collision
          first_collision.first_unit.current_checkpoint += 1
          reward += self._get_checkpoint_visit_reward(car_index=car_index)

        t += first_collision.time

    if self.align_reward_beta > 0.0:
      cp_x, cp_y = self.checkpoints[target_cp_idx]
      d_vec = np.array([cp_x - self.car.x, cp_y - self.car.y], dtype=np.float64)
      d_norm = np.linalg.norm(d_vec)
      d = d_vec / d_norm if d_norm > 0 else np.zeros(2)

      v_vec = np.array([self.car.vx, self.car.vy], dtype=np.float64)
      v_norm = np.linalg.norm(v_vec)
      v_normalized = v_vec / v_norm if v_norm > 0 else np.zeros(2)

      R_align = self.align_reward_beta * np.dot(v_normalized, d)
      reward += R_align

    return reward

  def _load_car_img(self) -> None:
    super()._load_car_img()
    self.opponent_car_img = (
        self._load_img(
            filename=self.opponent_car_img_path,
            width=self.checkpoint_radius,
        )
        if self.opponent_car
        else None
    )

  def _draw_car(self, canvas: pygame.Surface) -> None:
    for i, car in enumerate(self.cars):
      color = (230, 50, 50) if i == 0 else (50, 50, 230)
      # Solid body
      pygame.draw.circle(
          surface=canvas,
          color=color,
          center=[car.x / SCALE_FACTOR, car.y / SCALE_FACTOR],
          radius=self.car_radius / SCALE_FACTOR,
      )
      # Black outline
      pygame.draw.circle(
          surface=canvas,
          color=(0, 0, 0),
          center=[car.x / SCALE_FACTOR, car.y / SCALE_FACTOR],
          radius=self.car_radius / SCALE_FACTOR,
          width=2,
      )
      # Heading line
      angle_rad = car.radians()
      end_x = car.x / SCALE_FACTOR + (self.car_radius / SCALE_FACTOR) * np.cos(
          angle_rad
      )
      end_y = car.y / SCALE_FACTOR + (self.car_radius / SCALE_FACTOR) * np.sin(
          angle_rad
      )
      pygame.draw.line(
          surface=canvas,
          color=(255, 255, 255),
          start_pos=[car.x / SCALE_FACTOR, car.y / SCALE_FACTOR],
          end_pos=[end_x, end_y],
          width=3,
      )


class MadPodRacingBlockerEnv(MadPodRacingEnv):

  def __init__(
      self,
      opponent_path: str | Path,
      render_mode: str | None = None,
      laps: int = 3,
      car_max_thrust: int = 200,
      test_id: int | None = None,
      sequential_maps: bool = False,
      boost_on_first_move: bool = False,
      boost_opponent_on_first_move: bool = False,
      discover_checkpoints: bool = True,
      thrust_mapping: str = "linear",
  ) -> None:
    super().__init__(
        render_mode=render_mode,
        laps=laps,
        car_max_thrust=car_max_thrust,
        opponent_path=opponent_path,
        test_id=test_id,
        sequential_maps=sequential_maps,
        boost_on_first_move=boost_on_first_move,
        boost_opponent_on_first_move=boost_opponent_on_first_move,
        discover_checkpoints=discover_checkpoints,
        thrust_mapping=thrust_mapping,
    )

    # opponent runner observation, blocker car
    self.observation_space = spaces.Box(
        low=-1,
        high=1,
        shape=(16,),
        dtype=np.float64,
    )

  def _get_obs(self) -> ObsType:
    return self._get_blocker_obs(car_index=0)

  def _get_opponent_obs(self) -> ObsType:
    return self._get_runner_obs(car_index=1)

  def _get_collision_reward(self) -> SupportsFloat:
    return 0.1

  def _get_checkpoint_visit_reward(self, car_index: int) -> SupportsFloat:
    return -1 if car_index == 1 else 0


class MadPodRacingDiscreteEnv(MadPodRacingEnv):

  def __init__(
      self,
      render_mode: str | None = None,
      laps: int = 3,
      car_max_thrust: int = 200,
      test_id: int | None = None,
      sequential_maps: bool = False,
      opponent_path: str | Path | None = None,
      boost_on_first_move: bool = False,
      boost_opponent_on_first_move: bool = False,
      discover_checkpoints: bool = True,
      thrust_mapping: str = "linear",
      pruned_actions: bool = False,
      smart_thrust_cap: bool = False,
      align_reward_beta: float = 0.0,
      use_action_history: bool = False,
      smoothness_weight: float = 0.0,
      potential_alpha: float = 0.0,
      non_uniform_grid: int = 0,
      action_history_len: int = 0,
      use_asymmetric_history: bool = False,
      use_damped_thrust: bool = False,
      damped_thrust_beta: float = 0.9,
      thrust_levels: int = 2,
      custom_mid_thrust: int | None = None,
  ) -> None:
    super().__init__(
        render_mode=render_mode,
        laps=laps,
        car_max_thrust=car_max_thrust,
        test_id=test_id,
        sequential_maps=sequential_maps,
        opponent_path=opponent_path,
        boost_on_first_move=boost_on_first_move,
        boost_opponent_on_first_move=boost_opponent_on_first_move,
        discover_checkpoints=discover_checkpoints,
        thrust_mapping=thrust_mapping,
        pruned_actions=pruned_actions,
        smart_thrust_cap=smart_thrust_cap,
        align_reward_beta=align_reward_beta,
        use_action_history=use_action_history,
        smoothness_weight=smoothness_weight,
        potential_alpha=potential_alpha,
        non_uniform_grid=non_uniform_grid,
        action_history_len=action_history_len,
        use_asymmetric_history=use_asymmetric_history,
        use_damped_thrust=use_damped_thrust,
        damped_thrust_beta=damped_thrust_beta,
    )

    if self.pruned_actions:
      if self.non_uniform_grid == 7:
        angles = [-18, -6, -2, 0, 2, 6, 18]
      elif self.non_uniform_grid == 9:
        angles = [-18, -9, -4, -1, 0, 1, 4, 9, 18]
      elif self.non_uniform_grid == 13:
        angles = [-18, -12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12, 18]
      else:
        angles = [-18, -9, 0, 9, 18]
    else:
      angles = list(
          range(
              -self.max_rotation_per_turn,
              self.max_rotation_per_turn + 1,
          )
      )

    if thrust_levels == 3:
      mid = (
          custom_mid_thrust
          if custom_mid_thrust is not None
          else self.car_max_thrust // 2
      )
      thrusts = [0, mid, self.car_max_thrust]
    else:
      thrusts = [0, self.car_max_thrust]

    self.actions = list(product(angles, thrusts))
    self.action_space = spaces.Discrete(len(self.actions))

  def _convert_action_to_angle_thrust(
      self,
      action: ActType,
  ) -> tuple[float, float]:
    return self.actions[action]


class MadPodRacingBlockerDiscreteEnv(MadPodRacingBlockerEnv):

  def __init__(
      self,
      opponent_path: str | Path,
      render_mode: str | None = None,
      laps: int = 3,
      car_max_thrust: int = 200,
      test_id: int | None = None,
      sequential_maps: bool = False,
      boost_on_first_move: bool = False,
      boost_opponent_on_first_move: bool = False,
      discover_checkpoints: bool = True,
      thrust_mapping: str = "linear",
  ) -> None:
    super().__init__(
        opponent_path=opponent_path,
        render_mode=render_mode,
        laps=laps,
        car_max_thrust=car_max_thrust,
        test_id=test_id,
        sequential_maps=sequential_maps,
        boost_on_first_move=boost_on_first_move,
        boost_opponent_on_first_move=boost_opponent_on_first_move,
        discover_checkpoints=discover_checkpoints,
        thrust_mapping=thrust_mapping,
    )

    self.actions = list(
        product(
            list(
                range(
                    -self.max_rotation_per_turn,
                    self.max_rotation_per_turn + 1,
                )
            ),
            [0, self.car_max_thrust],
        )
    )
    self.action_space = spaces.Discrete(len(self.actions))

  def _convert_action_to_angle_thrust(
      self,
      action: ActType,
  ) -> tuple[float, float]:
    return self.actions[action]
