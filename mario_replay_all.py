import json
import sys
import time
import glob
import os
import pygame as pg

sys.path.insert(0, ".")
from mario_env import MarioEnv
from leaderboard import save_score, print_leaderboard

PAUSA_ENTRE_RUNS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5

archivos = sorted(glob.glob("checkpoints/*_acciones.json"))

if not archivos:
    print("No se encontraron archivos en checkpoints/*_acciones.json")
    print("Entrena primero con mario_env.py para generar checkpoints.")
    sys.exit()

print(f"\n{'='*50}")
print(f"  Reproduciendo {len(archivos)} checkpoints en orden")
print(f"{'='*50}\n")
for a in archivos:
    print(f"  {a}")
print()

env  = MarioEnv(render=True)
font = pg.font.SysFont("monospace", 16)
clock = pg.time.Clock()

def draw_hud(screen, archivo, num, total_archivos, frame, total_frames, mario_x, elapsed):
    nombre = os.path.basename(archivo).replace("_acciones.json", "")
    lines = [
        f"Run {num}/{total_archivos}",
        f"{nombre}",
        f"Frame: {frame}/{total_frames}",
        f"X:     {mario_x:.0f}",
        f"Tiempo:{elapsed:.1f}s",
    ]
    overlay = pg.Surface((240, len(lines)*22+10), pg.SRCALPHA)
    overlay.fill((0,0,0,180))
    screen.blit(overlay, (5,5))
    for i, line in enumerate(lines):
        color = (255,215,0) if i == 0 else (0,255,200) if i == 1 else (255,255,255)
        screen.blit(font.render(line, True, color), (10, 10+i*22))

def pantalla_transicion(screen, texto, subtexto=""):
    screen.fill((0,0,0))
    f_big   = pg.font.SysFont("monospace", 28, bold=True)
    f_small = pg.font.SysFont("monospace", 18)
    t1 = f_big.render(texto,    True, (255,215,0))
    t2 = f_small.render(subtexto, True, (180,180,180))
    screen.blit(t1, (800//2 - t1.get_width()//2, 250))
    screen.blit(t2, (800//2 - t2.get_width()//2, 300))
    pg.display.flip()

resultados = []

for num, archivo in enumerate(archivos, 1):
    with open(archivo) as f:
        actions = json.load(f)

    nombre = os.path.basename(archivo).replace("_acciones.json","")
    print(f"[{num}/{len(archivos)}] {nombre} — {len(actions)} frames")

    pantalla_transicion(
        env.screen,
        f"RUN {num}/{len(archivos)}",
        nombre
    )
    time.sleep(1.0)

    obs, _     = env.reset()
    start_time = time.time()
    resultado  = None

    for i, action in enumerate(actions):
        obs, reward, done, truncated, info = env.step(action)

        elapsed = time.time() - start_time
        pg.display.flip()
        clock.tick(60)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                print_leaderboard()
                env.close()
                sys.exit()

        if done or truncated:
            won = info['won']
            x   = info['mario_x']
            resultado = {"nombre": nombre, "x": x, "tiempo": elapsed, "won": won}
            resultados.append(resultado)
            save_score("IA", x, elapsed, won, is_ai=True)
            print(f"       X={x:.0f} | Tiempo={elapsed:.1f}s | {'¡Gano!  ' if won else 'Murio'}")
            break

    if resultado:
        pantalla_transicion(
            env.screen,
            f"X={resultado['x']:.0f}  {elapsed:.1f}s",
            "¡GANO!" if resultado['won'] else "Murio"
        )
        time.sleep(PAUSA_ENTRE_RUNS)
print(f"\n{'='*50}")
print("  RESUMEN DE EVOLUCION")
print(f"{'='*50}")
for i, r in enumerate(resultados, 1):
    barra = "█" * int(r['x'] / 100)
    print(f"  Run {i:>2} | {r['nombre'][-20:]:<20} | X={r['x']:>5.0f} {barra}")

print()
print_leaderboard()
env.close()