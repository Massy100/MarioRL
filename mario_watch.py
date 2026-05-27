import sys
import time
import pygame as pg
from stable_baselines3 import PPO
from mario_env import MarioEnv
from leaderboard import save_score, print_leaderboard

model_path = sys.argv[1] if len(sys.argv) > 1 else "best_model"

env   = MarioEnv(render=True)
model = PPO.load(model_path, env=env)

font = pg.font.SysFont("monospace", 16)

def draw_hud(screen, action, keys, jump_hold, mario_x, elapsed):
    action_names = {0:"quieto", 1:"caminar", 2:"correr", 3:"cam+salto", 4:"cor+salto"}
    lines = [
        f"Accion:  {action} ({action_names.get(action,'?')})",
        f"RIGHT:   {'ON' if keys.get(pg.K_RIGHT) else 'off'}",
        f"K_a:     {'ON' if keys.get(pg.K_a)     else 'off'}",
        f"K_s:     {'ON' if keys.get(pg.K_s)     else 'off'}",
        f"hold:    {jump_hold}",
        f"X:       {mario_x:.0f}",
    ]
    overlay = pg.Surface((180, len(lines)*20+10), pg.SRCALPHA)
    overlay.fill((0,0,0,160))
    screen.blit(overlay, (5,5))
    for i, line in enumerate(lines):
        color = (0,255,0) if "ON" in line else (255,255,0) if "★" in line else (255,255,255)
        screen.blit(font.render(line, True, color), (10, 10+i*20))

obs, _ = env.reset()
total_r    = 0.0
start_time = time.time()
last_action = 0
last_keys   = {}

print(f"\nCargando modelo: {model_path}")
print("Cerrando ventana para salir\n")

while True:
    action, _ = model.predict(obs, deterministic=False)
    action = int(action)

    last_keys = env._action_to_keys.__wrapped__(env, action) if hasattr(env._action_to_keys, '__wrapped__') else {}

    obs, r, done, truncated, info = env.step(action)
    total_r += r

    elapsed = time.time() - start_time
    draw_hud(env.screen, action, last_keys,
             env.jump_hold_remaining, info['mario_x'], elapsed)
    pg.display.flip()

    if done or truncated:
        won = info['won']
        x   = info['mario_x']
        print(f"  X: {x:.0f} | Tiempo: {elapsed:.1f}s | "
              f"Reward: {total_r:.1f} | {'¡GANÓ!' if won else 'Murió'}")

        ai_name = "AI " + model_path[-3:].upper().replace("EL","EL")
        save_score(" IA", x, elapsed, won, is_ai=True)

        obs, _ = env.reset()
        total_r    = 0.0
        start_time = time.time()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            print_leaderboard()
            env.close()
            sys.exit()