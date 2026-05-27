import json
import os
import sys
import numpy as np
import pygame as pg
import gymnasium as gym
from gymnasium import spaces


sys.path.insert(0, ".")
from data import setup, tools, constants as c, game_sound
from data.states.level1 import Level1



FLAGPOLE_X      = 8505    
SCREEN_W        = 800
SCREEN_H        = 600
MAX_STEPS       = 20_000  
JUMP_HOLD_FRAMES = 18     
N_ACTIONS = 5


class MarioEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render: bool = False):
        super().__init__()

        self.render_mode = "human" if render else None
        self._render = render

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        if not pg.get_init():
            pg.init()

        if self._render:
            self.screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
            pg.display.set_caption("Mario RL")
        else:
            self.screen = pg.Surface((SCREEN_W, SCREEN_H))

        self.level               = None
        self.clock               = pg.time.Clock()
        self.step_count          = 0
        self.last_x              = 0
        self.max_x               = 0   
        self.stale_timer         = 0
        self.jump_hold_remaining = 0

        self._load_game_assets()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.level               = self._create_fresh_level()
        self.step_count          = 0
        self.last_x              = self.level.mario.rect.x
        self.max_x               = self.level.mario.rect.x
        self.stale_timer         = 0
        self.jump_hold_remaining = 0
        return self._get_obs(), {}

    def step(self, action: int):
        keys = self._action_to_keys(action)

        prev_count = len(list(self.level.enemy_group))

        current_time = pg.time.get_ticks()
        self.level.update(self.screen, keys, current_time)

        curr_count = len(list(self.level.enemy_group))

        if self._render:
            self._draw_debug_overlay(action, keys)
            pg.display.flip()  
            self.clock.tick(self.metadata["render_fps"])
        else:
            pg.event.pump()

        self.step_count += 1

        mario_x    = self.level.mario.rect.x
        mario_dead = self.level.mario.dead
        won        = self.level.mario.in_castle or (mario_x >= FLAGPOLE_X)

        reward = self._compute_reward(mario_x, mario_dead, won)

        terminated = mario_dead or won
        truncated  = self.step_count >= MAX_STEPS

        self.last_x = mario_x
        self.max_x  = max(self.max_x, mario_x)

        info = {
            "mario_x":    mario_x,
            "max_x":      self.max_x,
            "step_count": self.step_count,
            "won":        won,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return pg.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self):
        pg.quit()

    def _load_game_assets(self):
        try:
            if not setup.GFX:
                setup.GFX.update(tools.load_all_gfx("resources/graphics"))
            if not setup.SFX:
                setup.SFX.update(tools.load_all_sfx("resources/sound"))
            if not setup.FONTS:
                setup.FONTS.update(tools.load_all_fonts("resources/fonts"))
        except Exception as e:
            print(f"[MarioEnv] Advertencia al cargar assets: {e}")

    def _create_fresh_level(self) -> Level1:
        game_info = {
            c.COIN_TOTAL:   0,
            c.SCORE:        0,
            c.TOP_SCORE:    0,
            c.LIVES:        3,
            c.CURRENT_TIME: 0,
            c.LEVEL_STATE:  c.NOT_FROZEN,
            c.MARIO_DEAD:   False,
            c.CAMERA_START_X: 0,
        }
        level = Level1()
        level.startup(pg.time.get_ticks(), game_info)
        setup.SCREEN = self.screen
        return level

    def _action_to_keys(self, action: int) -> dict:
        keys = {k: False for k in
                [pg.K_LEFT, pg.K_RIGHT, pg.K_a, pg.K_s, pg.K_UP, pg.K_DOWN]}

        if action == 0:
            pass                              
        elif action == 1:
            keys[pg.K_RIGHT] = True          
        elif action == 2:
            keys[pg.K_RIGHT] = True           
            keys[pg.K_s]     = True
        elif action == 3:                    
            keys[pg.K_RIGHT] = True
            if self.jump_hold_remaining == 0:  
                self.jump_hold_remaining = JUMP_HOLD_FRAMES
        elif action == 4:                     
            keys[pg.K_RIGHT] = True
            keys[pg.K_s]     = True
            if self.jump_hold_remaining == 0:  
                self.jump_hold_remaining = JUMP_HOLD_FRAMES

        if self.jump_hold_remaining > 0:
            keys[pg.K_a] = False  
            keys[pg.K_a] = True
            self.jump_hold_remaining -= 1
        else:
            keys[pg.K_a] = False

        return keys

    def _get_obs(self) -> np.ndarray:
        mario  = self.level.mario
        m_x    = float(mario.rect.x)
        m_y    = float(mario.rect.y)
        m_xvel = float(getattr(mario, "x_vel", 0))
        m_yvel = float(getattr(mario, "y_vel", 0))

        on_ground = 1.0 if mario.state not in (c.JUMP, c.FALL) else 0.0

        active_enemies = list(self.level.enemy_group)
        checkpoints    = list(self.level.check_point_group)
        all_threats    = active_enemies + checkpoints

        obstacles = list(self.level.pipe_group) + list(self.level.step_group)

        dist_enemy, h_enemy = self._nearest_ahead(mario, all_threats, max_dist=1000)
        dist_obs,   h_obs   = self._nearest_ahead(mario, obstacles,   max_dist=600)

        level_width = 8800.0
        return np.array([
            np.clip(m_x           / level_width, 0, 1),
            np.clip(m_y           / SCREEN_H,    0, 1),
            np.clip((m_xvel + 10) / 20,          0, 1),
            np.clip((m_yvel + 20) / 40,          0, 1),
            on_ground,
            dist_enemy / 600.0,
            h_enemy    / SCREEN_H,
            dist_obs   / 600.0,
            h_obs      / SCREEN_H,
        ], dtype=np.float32)

    def _nearest_ahead(self, mario, sprites, max_dist=600):
        best_dist, best_h = max_dist, 0.0
        mario_right = mario.rect.right
        mario_cy    = mario.rect.centery
        for sp in sprites:
            dx = sp.rect.left - mario_right
            if 0 < dx < best_dist:
                best_dist = dx
                best_h    = float(abs(sp.rect.top - mario_cy))
        return best_dist, best_h

    def _compute_reward(self, mario_x: float, dead: bool, won: bool) -> float:
        
        if won:
            return 500.0

        if dead:
            return -15.0

        advance = mario_x - self.last_x
        reward  = max(advance, 0) * 0.1  

        reward += 0.1                      
        if advance <= 0:
            self.stale_timer += 1
        else:
            self.stale_timer = 0

        if self.stale_timer > 30:
            reward -= 1.0

        if mario_x > 900 and self.last_x <= 900:
            reward += 200.0

        return reward

    def _draw_debug_overlay(self, action: int, keys: dict):
        if not self._render:
            return

        action_names = {
            0: "quieto",
            1: "caminar",
            2: "correr",
            3: "caminar+saltar",
            4: "correr+saltar"
        }

        font = pg.font.SysFont("monospace", 16)
        lines = [
            f"Accion: {action} ({action_names.get(action, '?')})",
            f"RIGHT:  {'ON' if keys.get(pg.K_RIGHT) else 'off'}",
            f"K_a:    {'ON' if keys.get(pg.K_a)     else 'off'}",
            f"K_s:    {'ON' if keys.get(pg.K_s)     else 'off'}",
            f"hold:   {self.jump_hold_remaining}",
            f"X:      {self.level.mario.rect.x}",
            f"state:  {self.level.mario.state}",
        ]

        overlay = pg.Surface((160, len(lines) * 20 + 10), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (5, 5))

        for i, line in enumerate(lines):
            color = (0, 255, 0) if "ON" in line else (255, 255, 255)
            text = font.render(line, True, color)
            self.screen.blit(text, (10, 10 + i * 20))

if __name__ == "__main__":
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import BaseCallback
    

    MODEL_PATH = "mario_ppo_model"

    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.best_actions = []
            self.current_actions = []
            self.ep_reward = 0.0
            self.best_x = 0

        def _on_step(self) -> bool:

            self.ep_reward += self.locals["rewards"][0]
            self.current_actions.append(int(self.locals["actions"][0]))
            if self.locals["dones"][0]:
                self.episode_rewards.append(self.ep_reward)
                info = self.locals["infos"][0]
                x    = info.get("mario_x", 0)

                
                if len(self.episode_rewards) % 10 == 0:
                    avg = np.mean(self.episode_rewards[-50:])
                    print(f"  Ep {len(self.episode_rewards):>4} | X: {x:>5.0f} | "
                        f"Mejor X: {self.best_x:>5.0f} | "
                        f"Reward: {self.ep_reward:>7.1f} | Avg50: {avg:>7.1f}")
                    
                if x > self.best_x:
                    self.best_x = x
                    ep = len(self.episode_rewards)
                    nombre = f"checkpoints/mario_x{int(x)}_ep{ep}"
                    try:
                        import os
                        os.makedirs("checkpoints", exist_ok=True)
                        self.model.save(nombre)
                        self.model.save("best_model") 
                        print(f"    Nuevo mejor: X={x:.0f} — guardado como {nombre}")
                        self.best_actions = self.current_actions.copy()
                        with open(f"{nombre}_acciones.json", "w") as f:
                            json.dump(self.best_actions, f)
                    except Exception as e:
                        print(f"  ERROR al guardar: {e}")
                
                if info["won"]:
                    with open("best_run.json", "w") as f:
                        json.dump(self.best_actions, f)
                    print(f"    Secuencia de mejores acciones guardada ({len(self.best_actions)} pasos)")
                
                self.ep_reward = 0.0
                self.current_actions = []

            return True



    print("=" * 60)
    print("  Mario RL — PPO")
    print("=" * 60)

    env = MarioEnv(render=False)
    check_env(env, warn=True)

    print("\n Verificando entorno...")
    check_env(env, warn=True)
    print("  ✓ OK\n")

    print(" Preparando modelo PPO...")
    if os.path.exists(f"{MODEL_PATH}.zip"):
        print(f"  Cargando modelo anterior: {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("  Creando modelo nuevo...")
        model = PPO(
            policy           = "MlpPolicy",
            env              = env,
            learning_rate    = 3e-4,
            n_steps          = 2048,   
            batch_size       = 64,
            n_epochs         = 10,
            gamma            = 0.99,
            gae_lambda       = 0.95,
            clip_range       = 0.2,
            ent_coef         = 0.01,   
            verbose          = 0,
            tensorboard_log  = "./mario_tensorboard/",
        )
    print("    Listo\n")

    TOTAL_STEPS = 500_000
    print(f"Entrenando {TOTAL_STEPS:,} pasos...")
    print("  (Ctrl+C para detener y guardar)\n")

    try:
        model.learn(
            total_timesteps     = TOTAL_STEPS,
            callback            = ProgressCallback(),
            progress_bar        = True,
            reset_num_timesteps = False,
        )
    except KeyboardInterrupt:
        print("\n  Detenido manualmente.")

    model.save(MODEL_PATH)
    print(f"\n    Modelo guardado: {MODEL_PATH}.zip")
    env.close()

    print("\n  Abriendo visualizacion...")
    env_vis = MarioEnv(render=True)
    model   = PPO.load("best_model", env=env_vis)

    obs, _ = env_vis.reset()
    total_r = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, truncated, info = env_vis.step(int(action))
        total_r += r
        if done or truncated:
            print(f"  X: {info['mario_x']:.0f} | "
                  f"Mejor X ep: {info['max_x']:.0f} | "
                  f"Reward: {total_r:.1f} | "
                  f"{'¡Gano!' if info['won'] else 'Murio'}")
            obs, _ = env_vis.reset()
            total_r = 0.0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                env_vis.close()
                sys.exit()