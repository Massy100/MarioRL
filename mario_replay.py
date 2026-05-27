import json
import sys
import time
import pygame as pg

sys.path.insert(0, ".")
from mario_env import MarioEnv
from leaderboard import save_score, print_leaderboard

actions_file = sys.argv[1] if len(sys.argv) > 1 else "best_run_win.json"

print(f"\nReproduciendo: {actions_file}")

with open(actions_file) as f:
    actions = json.load(f)

print(f"Total de frames: {len(actions)}")

env  = MarioEnv(render=True)
font = pg.font.SysFont("monospace", 16)


obs, _     = env.reset()
start_time = time.time()

for i, action in enumerate(actions):
    obs, reward, done, truncated, info = env.step(action)

    elapsed = time.time() - start_time
    pg.display.flip()

    for event in pg.event.get():
        if event.type == pg.QUIT:
            env.close()
            sys.exit()

    if done or truncated:
        won = info['won']
        x   = info['mario_x']
        print(f"\nResultado: X={x:.0f} | Tiempo={elapsed:.1f}s | {'¡Ganó!' if won else 'Murió'}")

        name = actions_file.split("/")[-1][:3].upper().replace("_","R").replace("B","BST")
        save_score("RPL", x, elapsed, won, is_ai=True)
        print(f"Score guardado en leaderboard como RPL")
        print_leaderboard()
        break

env.close()