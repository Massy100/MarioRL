import json
import os
from datetime import datetime

LEADERBOARD_FILE = "leaderboard.json"

def load_scores():
    if not os.path.exists(LEADERBOARD_FILE):
        return []
    with open(LEADERBOARD_FILE) as f:
        return json.load(f)

def save_score(name: str, mario_x: float, time_secs: float, won: bool, is_ai: bool):
    scores = load_scores()
    scores.append({
        "name":    name.upper()[:3],
        "x":       round(mario_x),
        "time":    round(time_secs, 2),
        "won":     won,
        "is_ai":   is_ai,
        "date":    datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    scores.sort(key=lambda s: (not s["won"], s["time"] if s["won"] else -s["x"]))
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    return scores

def print_leaderboard():
    scores = load_scores()
    if not scores:
        print("No hay scores todavía.")
        return

    print("\n" + "═" * 58)
    print("      LEADERBOARD — HUMANO vs IA  ")
    print("═" * 58)
    print(f"  {'#':<3} {'NOM':<5} {'TIPO':<8} {'X':<6} {'TIEMPO':<10} {'ESTADO'}")
    print("─" * 58)

    for i, s in enumerate(scores, 1):
        tipo   = "IA  " if s["is_ai"] else "HUM "
        estado = "  GANO" if s["won"] else f"  X={s['x']}"
        tiempo = f"{s['time']:.1f}s" if s["won"] else "---"
        print(f"  {i:<3} {s['name']:<5} {tipo:<8} {s['x']:<6} {tiempo:<10} {estado}")

    print("═" * 58 + "\n")

print_leaderboard()
