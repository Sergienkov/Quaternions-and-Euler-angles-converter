# convert_camm_orientation.py
# -*- coding: utf-8 -*-
import argparse, json, math, os, csv
from math import copysign
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt  # only if --plots is used
except Exception:
    plt = None


# -------------------- Кватернионы и математика --------------------
def q_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n == 0.0 or not np.isfinite(n):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def integrate_gyro(q: np.ndarray, gyro_b: np.ndarray, dt: float) -> np.ndarray:
    """Интегрировать кватернион (корпус→мир) по гироскопу (рад/с) за dt"""
    if dt <= 0.0 or not np.isfinite(dt):
        return q
    omega = np.array([0.0, gyro_b[0], gyro_b[1], gyro_b[2]], dtype=float)
    qdot = 0.5 * q_mul(q, omega)
    q_new = q + qdot * dt
    return q_normalize(q_new)


def accel_feedback(q: np.ndarray, accel_b: np.ndarray,
                   Kp: float, Ki: float, e_int: np.ndarray, dt: float,
                   use_gate: bool = True,
                   g_min: float = 0.4, g_max: float = 2.0) -> (np.ndarray, np.ndarray):
    """Возвратить корректировку по акселю (вектор гравитации в корпусе) и обновлённый интеграл."""
    a = np.array(accel_b, dtype=float)
    norm_a = float(np.linalg.norm(a))
    if norm_a < 1e-9 or not np.isfinite(norm_a):
        return np.zeros(3, dtype=float), e_int

    # Нормируем аксель; опциональная «шлюзовая» проверка по g
    g_ratio = norm_a / 9.80665 if norm_a > 0 else 0.0
    if use_gate and not (g_min <= g_ratio <= g_max):
        return np.zeros(3, dtype=float), e_int

    a_norm = a / norm_a  # измеренный юнит-вектор гравитации в корпусе

    # Оценённый юнит-вектор гравитации в корпусе из q (корпус→мир)
    w, x, y, z = q
    vx = 2.0*(x*z - w*y)
    vy = 2.0*(w*x + y*z)
    vz = w*w - x*x - y*y + z*z
    v = np.array([vx, vy, vz], dtype=float)

    # Ошибка = a_meas × a_est
    e = np.cross(a_norm, v)

    # Интегральная часть (если включена)
    if Ki > 0.0 and dt > 0.0 and np.isfinite(dt):
        e_int = e_int + e * dt

    fb = Kp * e + Ki * e_int
    return fb, e_int


def quat_to_euler_zyx(q: np.ndarray) -> (float, float, float):
    """Вернуть yaw, pitch, roll (ZYX) из кватерниона (корпус→мир)."""
    w, x, y, z = q
    # roll (x)
    sinr_cosp = 2.0*(w*x + y*z)
    cosr_cosp = 1.0 - 2.0*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y)
    sinp = 2.0*(w*y - z*x)
    if abs(sinp) >= 1.0:
        pitch = copysign(math.pi/2.0, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (z)
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw, pitch, roll


def euler_zyx_to_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Обратное преобразование: yaw-pitch-roll (ZYX) → кватернион (корпус→мир)."""
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    # порядок Z * Y * X
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return q_normalize(np.array([w, x, y, z], dtype=float))


def wrap_to_pi(angle: float) -> float:
    """Сжать в (-pi, pi]."""
    a = (angle + math.pi) % (2.0 * math.pi) - math.pi
    # Нормируем +pi в -pi для стабильности
    if a == -math.pi:
        return math.pi
    return a


# -------------------- Парсинг camm.json и обработка --------------------
def load_events_camm(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = []
    for e in data:
        t = e.get("time")
        if t is None:
            continue
        # Приводим к int наносекундам
        try:
            t_int = int(t)
        except Exception:
            t_int = int(float(t))
        typ = e.get("type", None)
        if typ in (2, 3):  # 2=gyro, 3=accel
            events.append({
                "t": t_int,
                "type": int(typ),
                "gyro": e.get("gyro"),
                "accel": e.get("acceleration"),
            })
    events.sort(key=lambda x: x["t"])
    return events


def process_orientation(events: List[Dict[str, Any]],
                        gyro_units: str = "rad",
                        kp: float = 0.5, ki: float = 0.0,
                        accel_gate_min_g: float = 0.4,
                        accel_gate_max_g: float = 2.0,
                        unwrap_yaw: bool = False) -> List[Dict[str, Any]]:
    if not events:
        return []

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # корпус→мир
    last_gyro = np.zeros(3, dtype=float)
    e_int = np.zeros(3, dtype=float)

    t0 = events[0]["t"]
    t_prev = t0

    rows = []
    last_yaw = None
    yaw_unw = 0.0

    unit_scale = math.pi / 180.0 if gyro_units.lower().startswith("deg") else 1.0

    for ev in events:
        t = ev["t"]
        dt = (t - t_prev) / 1e9
        if dt < 0:
            dt = 0.0

        gyro_eff = last_gyro.copy()

        # Коррекция по акселю (если доступна)
        if ev["type"] == 3 and ev["accel"] is not None:
            fb, e_int = accel_feedback(q, ev["accel"], kp, ki, e_int, dt,
                                       use_gate=True,
                                       g_min=accel_gate_min_g,
                                       g_max=accel_gate_max_g)
            gyro_eff = gyro_eff + fb

        # Интеграция
        q = integrate_gyro(q, gyro_eff, dt)
        t_prev = t

        # Обновить гироскоп (сырые единицы → рад/с)
        if ev["type"] == 2 and ev["gyro"] is not None:
            g = np.array(ev["gyro"], dtype=float) * unit_scale
            last_gyro = g

        # Эйлеровы углы
        yaw, pitch, roll = quat_to_euler_zyx(q)

        # Развёрнутый yaw
        if unwrap_yaw:
            if last_yaw is None:
                yaw_unw = yaw
            else:
                delta = wrap_to_pi(yaw - last_yaw)
                yaw_unw = yaw_unw + delta
            last_yaw = yaw

        row = {
            "time_ns": int(t),
            "t_sec": (t - t0) / 1e9,
            "qw": float(q[0]), "qx": float(q[1]), "qy": float(q[2]), "qz": float(q[3]),
            "yaw_rad": float(yaw), "pitch_rad": float(pitch), "roll_rad": float(roll),
            "yaw_deg": math.degrees(yaw), "pitch_deg": math.degrees(pitch), "roll_deg": math.degrees(roll),
        }
        if unwrap_yaw:
            row["yaw_unwrapped_rad"] = float(yaw_unw)
            row["yaw_unwrapped_deg"] = math.degrees(yaw_unw)

        rows.append(row)

    return rows


def save_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        # создаём пустой файл с заголовком
        headers = ["time_ns","t_sec","qw","qx","qy","qz",
                   "yaw_rad","pitch_rad","roll_rad","yaw_deg","pitch_deg","roll_deg",
                   "yaw_unwrapped_rad","yaw_unwrapped_deg"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)
        return

    headers = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_plots(rows: List[Dict[str, Any]], out_dir: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib не установлен. Установите: pip install matplotlib")
    os.makedirs(out_dir, exist_ok=True)

    t = np.array([r["t_sec"] for r in rows], dtype=float)
    roll = np.array([r["roll_deg"] for r in rows], dtype=float)
    pitch = np.array([r["pitch_deg"] for r in rows], dtype=float)
    yaw = np.array([r["yaw_deg"] for r in rows], dtype=float)

    def plot_one(x, y, xlab, ylab, title, fn):
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(title)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fn), dpi=140)
        plt.close()

    plot_one(t, roll, "t, s", "roll, deg", "Roll vs time", "roll_vs_time.png")
    plot_one(t, pitch, "t, s", "pitch, deg", "Pitch vs time", "pitch_vs_time.png")
    plot_one(t, yaw, "t, s", "yaw, deg", "Yaw vs time", "yaw_vs_time.png")


# -------------------- CLI --------------------
def main():
    p = argparse.ArgumentParser(description="CAMM IMU → Quaternion + Euler (ZYX) CSV")
    p.add_argument("-i", "--input", required=True, help="Путь к camm.json")
    p.add_argument("-o", "--output", required=True, help="Путь к CSV с ориентацией")
    p.add_argument("--gyro-units", choices=["rad","deg"], default="rad",
                   help="Единицы гироскопа во входных данных (по умолчанию rad/s)")
    p.add_argument("--kp", type=float, default=0.5, help="Kp фильтра Махони (по умолчанию 0.5)")
    p.add_argument("--ki", type=float, default=0.0, help="Ki фильтра Махони (по умолчанию 0.0)")
    p.add_argument("--accel-gate-min", type=float, default=0.4, help="Мин. модуль |a|/g для коррекции (по умолчанию 0.4)")
    p.add_argument("--accel-gate-max", type=float, default=2.0, help="Макс. модуль |a|/g для коррекции (по умолчанию 2.0)")
    p.add_argument("--unwrap-yaw", action="store_true", help="Разворачивать yaw (непрерывная фаза)")
    p.add_argument("--plots", default=None, help="Папка для PNG‑графиков (опционально)")
    args = p.parse_args()

    events = load_events_camm(args.input)
    if not events:
        raise SystemExit("IMU-события type=2/3 не найдены в входном файле.")

    rows = process_orientation(
        events,
        gyro_units=args.gyro_units,
        kp=args.kp, ki=args.ki,
        accel_gate_min_g=args.accel_gate_min,
        accel_gate_max_g=args.accel_gate_max,
        unwrap_yaw=args.unwrap_yaw
    )

    save_csv(rows, args.output)
    print(f"✔ CSV сохранён: {args.output}  (строк: {len(rows)})")

    if args.plots:
        make_plots(rows, args.plots)
        print(f"✔ Графики сохранены в: {args.plots}")


if __name__ == "__main__":
    main()
