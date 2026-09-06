import csv
import time
import serial
import wlkatapython

CSV_PATH = "추론_용접선_웨이포인트.csv"
TARGET_CONDITION = "WELD_D08_L1_A030"

START_X = -128.0
START_Y = 250.0
START_Z = 12.2

RX = 0.0
RY = 0.0
RZ = 0.0

SCALE = 0.85
SERIAL_PORT = "COM7"
BAUD_RATE = 115200
COMMAND_INTERVAL = 0.1

def load_waypoints(csv_path, condition):
    waypoints = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["condition"] != condition:
                continue
            waypoint = {
                "seq": int(row["seq"]),
                "x": float(row["x_local_mm"]),
                "y": float(row["y_local_mm"]),
                "z": float(row["z_local_mm"]),
            }
            waypoints.append(waypoint)
    waypoints.sort(key=lambda p: p["seq"])
    return waypoints

def convert_to_robot_coordinates(waypoints):
    robot_points = []
    if not waypoints:
        return robot_points
    x0, y0, z0 = waypoints[0]["x"], waypoints[0]["y"], waypoints[0]["z"]
    for p in waypoints:
        dx = (p["x"] - x0) * SCALE
        dy = (p["y"] - y0) * SCALE
        dz = (p["z"] - z0) * SCALE
        robot_x = round(START_Y - dy, 1)
        robot_y = round(START_X + dx, 1)
        robot_z = round(START_Z + dz, 1)
        robot_points.append({
            "seq": p["seq"],
            "x": robot_x,
            "y": robot_y,
            "z": robot_z,
        })
    return robot_points

def follow_waypoints(mirobot, waypoints):
    for p in waypoints:
        print(f"[{p['seq']:03d}] X={p['x']:.2f}, Y={p['y']:.2f}, Z={p['z']:.2f}")
        mirobot.writecoordinate(0, 0, p["x"], p["y"], p["z"], RX, RY, RZ)

def main():
    waypoints = load_waypoints(CSV_PATH, TARGET_CONDITION)
    serial1 = serial. Serial("COM7", 115200)
    mirobot1 = wlkatapython.Wlkata_UART()
    mirobot1.init(serial1,-1)

    print(f"{TARGET_CONDITION}: {len(waypoints)} waypoints loaded")

    if not waypoints:
        print("웨이포인트가 없습니다.")
        return

    print("\n=== WAYPOINTS ===")
    for p in waypoints:
        print(
            f"{p['seq']:03d}: "
            f"X={p['x']:.2f}, "
            f"Y={p['y']:.2f}, "
            f"Z={p['z']:.2f}"
        )

    robot_points = convert_to_robot_coordinates(waypoints)

    print("\n=== ROBOT WAYPOINTS ===")
    for p in robot_points:
        print(
            f"{p['seq']:03d}: "
            f"X={p['x']:.1f}, "
            f"Y={p['y']:.1f}, "
            f"Z={p['z']:.1f}"
        )

    follow_waypoints(mirobot1, robot_points)
    mirobot1.zero()


if __name__ == "__main__":
    main()