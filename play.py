import sys
import time
import json
import os
import pygame as pg

sys.path.insert(0, ".")
from mario_env import MarioEnv, FLAGPOLE_X
from leaderboard import save_score

pg.init()

env = MarioEnv(render=True)
pg.display.set_caption("Mario — Modo Humano")

font_big   = pg.font.SysFont("monospace", 28, bold=True)
font_small = pg.font.SysFont("monospace", 18)

def get_initials():
    screen = env.screen
    initials = []
    clock = pg.time.Clock()

    while True:
        screen.fill((0, 0, 0))
        title = font_big.render("INGRESA TUS INICIALES", True, (255, 200, 0))
        hint  = font_small.render("3 letras, Enter para confirmar", True, (180, 180, 180))
        shown = font_big.render("".join(initials) + ("_" if len(initials) < 3 else ""), True, (255, 255, 255))

        screen.blit(title,  (800//2 - title.get_width()//2,  180))
        screen.blit(hint,   (800//2 - hint.get_width()//2,   230))
        screen.blit(shown,  (800//2 - shown.get_width()//2,  300))
        pg.display.flip()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                env.close()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RETURN and len(initials) == 3:
                    return "".join(initials).upper()
                if event.key == pg.K_BACKSPACE and initials:
                    initials.pop()
                elif len(initials) < 3 and event.unicode.isalpha():
                    initials.append(event.unicode.upper())
        clock.tick(30)

def draw_hud(screen, elapsed, mario_x, won, dead):
    """HUD con tiempo y posición."""
    color = (0, 255, 0) if won else (255, 80, 80) if dead else (255, 255, 255)
    t = font_small.render(f"Tiempo: {elapsed:.1f}s  |  X: {mario_x:.0f}", True, color)
    screen.blit(t, (10, 10))

obs, _ = env.reset()
clock  = pg.time.Clock()

start_time = time.time()
mario_x    = 0
won        = False
dead       = False
running    = True

while running:
    keys_pressed = pg.key.get_pressed()

    right = keys_pressed[pg.K_RIGHT]
    left  = keys_pressed[pg.K_LEFT]
    jump  = keys_pressed[pg.K_a]
    run   = keys_pressed[pg.K_s]

    if jump and right and run:
        action = 4
    elif jump and right:
        action = 3
    elif run and right:
        action = 2
    elif right:
        action = 1
    else:
        action = 0

    env.level.update(env.screen, keys_pressed, pg.time.get_ticks())
    pg.display.flip()

    mario_x = env.level.mario.rect.x
    dead    = env.level.mario.dead
    won     = env.level.mario.in_castle or (mario_x >= FLAGPOLE_X)

    elapsed = time.time() - start_time
    draw_hud(env.screen, elapsed, mario_x, won, dead)
    pg.display.flip()

    if won or dead:
        running = False

    for event in pg.event.get():
        if event.type == pg.QUIT:
            env.close()
            sys.exit()

    clock.tick(60)

elapsed = time.time() - start_time
print(f"\nResultado: X={mario_x:.0f} | Tiempo={elapsed:.1f}s | {'¡Ganaste!' if won else 'Moriste'}")

initials = get_initials()
save_score(initials, mario_x, elapsed, won, is_ai=False)
print(f"Score guardado para {initials}!")

env.close()