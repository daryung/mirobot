import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_COLUMNS = {"condition", "seq"}

FRAME_COLUMNS = {
    "raw": ("x_raw_mm", "y_raw_mm", "z_raw_mm"),
    "cam": ("x_cam_mm", "y_cam_mm", "z_cam_mm"),
    "local": ("x_local_mm", "y_local_mm", "z_local_mm"),
}


def load_waypoints(csv_path: str) -> pd.DataFrame:
    """웨이포인트 CSV를 읽고 필수 열을 확인한다."""
    df = pd.read_csv(csv_path)

    required = set(BASE_COLUMNS)
    for cols in FRAME_COLUMNS.values():
        required.update(cols)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다:\n"
            + "\n".join(f"- {c}" for c in sorted(missing))
        )

    numeric_cols = ["seq"]
    for cols in FRAME_COLUMNS.values():
        numeric_cols.extend(cols)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["condition", "seq"]).copy()
    return df


def choose_condition(df: pd.DataFrame, requested: str | None) -> str:
    conditions = sorted(df["condition"].astype(str).unique())

    if not conditions:
        raise ValueError("CSV에 condition 데이터가 없습니다.")

    if requested is not None:
        if requested not in conditions:
            raise ValueError(
                f"'{requested}' 조건이 없습니다.\n"
                f"사용 가능한 조건 예: {conditions[:10]}"
            )
        return requested

    print("\n=== 사용 가능한 조건 ===")
    for i, condition in enumerate(conditions, start=1):
        print(f"{i:3d}. {condition}")

    while True:
        value = input("\n시각화할 번호 입력: ").strip()
        try:
            index = int(value) - 1
            if 0 <= index < len(conditions):
                return conditions[index]
        except ValueError:
            pass

        print("올바른 번호를 입력하세요.")


def choose_frame(requested: str | None) -> str:
    if requested is not None:
        return requested

    print("\n=== 좌표계 선택 ===")
    print("1. LOCAL  - 로봇 작업대 형상 확인용")
    print("2. RAW    - PLY 원본 좌표계")
    print("3. CAM    - RealSense 카메라 좌표계")
    print("4. ALL    - RAW / CAM / LOCAL 한 번에 비교")
    print("5. XY     - LOCAL 경로를 XY 2D 평면에서 보기")

    mapping = {
        "1": "local",
        "2": "raw",
        "3": "cam",
        "4": "all",
        "5": "xy",
    }

    while True:
        value = input("\n번호 입력: ").strip()
        if value in mapping:
            return mapping[value]
        print("1~5 중 하나를 입력하세요.")


def get_frame_data(part: pd.DataFrame, frame: str):
    x_col, y_col, z_col = FRAME_COLUMNS[frame]

    valid = part.dropna(subset=[x_col, y_col, z_col]).copy()

    if valid.empty:
        raise ValueError(f"{frame.upper()} 좌표에 유효한 데이터가 없습니다.")

    x = valid[x_col].to_numpy()
    y = valid[y_col].to_numpy()
    z = valid[z_col].to_numpy()
    seq = valid["seq"].astype(int).to_numpy()

    return valid, x, y, z, seq


def equal_3d_axes(ax, x, y, z):
    """
    X/Y/Z 축 스케일을 동일하게 맞춘다.
    축 비율 차이로 경로가 과장되어 보이는 것을 줄인다.
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
        1.0,
    )
    half = max_range / 2

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def draw_path(ax, part: pd.DataFrame, frame: str, show_numbers: bool):
    _, x, y, z, seq = get_frame_data(part, frame)

    # 원본 결과 그림과 동일한 축 배치:
    # 가로 = x, 깊이(사선) = z, 수직 = y
    ax.plot(x, z, y, linewidth=2, label="Waypoint path")
    ax.scatter(x, z, y, s=30, label="Waypoints")

    ax.scatter(
        [x[0]], [z[0]], [y[0]],
        s=90, marker="o", label="Start"
    )
    ax.scatter(
        [x[-1]], [z[-1]], [y[-1]],
        s=100, marker="X", label="End"
    )

    if show_numbers:
        for xi, yi, zi, si in zip(x, y, z, seq):
            ax.text(xi, zi, yi, f" {si}", fontsize=7)

    x_col, y_col, z_col = FRAME_COLUMNS[frame]

    ax.set_xlabel(f"{x_col} (mm)")
    ax.set_ylabel(f"{z_col} (mm)")
    ax.set_zlabel(f"{y_col} (mm)")
    ax.set_title(frame.upper())

    # 표시 축 순서가 (x, z, y)이므로 동일한 순서로 비율 보정
    equal_3d_axes(ax, x, z, y)

    # 원본 조건별 시각화와 유사한 초기 시점
    ax.view_init(elev=18, azim=-65)

    return x, y, z


def print_frame_stats(frame: str, x, y, z):
    print(f"\n[{frame.upper()}]")
    print(f"X 범위 : {x.min():.2f} ~ {x.max():.2f} mm")
    print(f"Y 범위 : {y.min():.2f} ~ {y.max():.2f} mm")
    print(f"Z 범위 : {z.min():.2f} ~ {z.max():.2f} mm")
    print(f"X 변화폭: {(x.max() - x.min()):.2f} mm")
    print(f"Y 변화폭: {(y.max() - y.min()):.2f} mm")
    print(f"Z 변화폭: {(z.max() - z.min()):.2f} mm")


def visualize_single(
    part: pd.DataFrame,
    condition: str,
    frame: str,
    show_numbers: bool,
):
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")

    x, y, z = draw_path(ax, part, frame, show_numbers)

    ax.set_title(
        f"{condition}\n"
        f"{frame.upper()} waypoint 3D path | {len(x)} points"
    )
    ax.legend()

    print("\n=== 선택 조건 ===")
    print(condition)
    print_frame_stats(frame, x, y, z)

    plt.tight_layout()
    plt.show()


def visualize_all(
    part: pd.DataFrame,
    condition: str,
    show_numbers: bool,
):
    fig = plt.figure(figsize=(18, 6))

    frames = ["raw", "cam", "local"]

    for i, frame in enumerate(frames, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        x, y, z = draw_path(ax, part, frame, show_numbers)
        print_frame_stats(frame, x, y, z)

        if i == 1:
            ax.legend(loc="best")

    fig.suptitle(
        f"{condition} | RAW vs CAM vs LOCAL",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()



def visualize_xy(
    part: pd.DataFrame,
    condition: str,
    show_numbers: bool,
):
    """LOCAL waypoint를 위에서 내려다본 XY 2D 평면으로 표시한다."""
    _, x, y, z, seq = get_frame_data(part, "local")

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(x, y, linewidth=2, label="Waypoint path")
    ax.scatter(x, y, s=35, label="Waypoints")
    ax.scatter([x[0]], [y[0]], s=100, marker="o", label="Start")
    ax.scatter([x[-1]], [y[-1]], s=110, marker="X", label="End")

    if show_numbers:
        for xi, yi, si in zip(x, y, seq):
            ax.annotate(
                str(si),
                (xi, yi),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(
        f"{condition}\n"
        f"LOCAL waypoint XY top view | {len(x)} points"
    )
    ax.set_xlabel("x_local_mm (mm)")
    ax.set_ylabel("y_local_mm (mm)")

    # 실제 XY 형상이 왜곡되지 않도록 1 mm : 1 mm 비율 유지
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend()

    print("\n=== LOCAL XY 2D ===")
    print(f"X 범위 : {x.min():.2f} ~ {x.max():.2f} mm")
    print(f"Y 범위 : {y.min():.2f} ~ {y.max():.2f} mm")
    print(f"X 변화폭: {(x.max() - x.min()):.2f} mm")
    print(f"Y 변화폭: {(y.max() - y.min()):.2f} mm")
    print("※ XY 축은 동일 스케일(1:1)로 표시됩니다.")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="AI 용접선 waypoint 3D 좌표계 시각화"
    )

    parser.add_argument(
        "csv",
        nargs="?",
        default="추론_용접선_웨이포인트.csv",
        help="웨이포인트 CSV 경로",
    )

    parser.add_argument(
        "--condition",
        "-c",
        default=None,
        help="예: WELD_D08_L1_A130",
    )

    parser.add_argument(
        "--frame",
        "-f",
        choices=["local", "raw", "cam", "all", "xy"],
        default=None,
        help="시각화 좌표계: local / raw / cam / all / xy",
    )

    parser.add_argument(
        "--no-number",
        action="store_true",
        help="웨이포인트 seq 번호 숨김",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {csv_path.resolve()}"
        )

    df = load_waypoints(str(csv_path))
    condition = choose_condition(df, args.condition)
    frame = choose_frame(args.frame)

    part = (
        df[df["condition"].astype(str) == condition]
        .sort_values("seq")
        .reset_index(drop=True)
    )

    if part.empty:
        raise ValueError(f"{condition} 데이터가 없습니다.")

    print("\n참고:")
    print("- RAW   : PLY 원본 좌표계")
    print("- CAM   : RealSense 좌표계")
    print("- LOCAL : 경로 시작점 원점 + 경로 평면 정렬")
    print("- 3D 표시축: 가로=X, 깊이(사선)=Z, 수직=Y (원본 조건별 PNG와 동일한 배치)")
    print("- 자료 기준 CAM = (x_raw, -y_raw, -z_raw)")

    if frame == "all":
        visualize_all(
            part,
            condition,
            show_numbers=not args.no_number,
        )
    elif frame == "xy":
        visualize_xy(
            part,
            condition,
            show_numbers=not args.no_number,
        )
    else:
        visualize_single(
            part,
            condition,
            frame,
            show_numbers=not args.no_number,
        )


if __name__ == "__main__":
    main()
