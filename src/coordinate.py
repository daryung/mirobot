from .config import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    ROBOT_X_MAX,
    ROBOT_X_MIN,
    ROBOT_Y_MAX,
    ROBOT_Y_MIN,
)


def canvas_to_robot(canvas_x, canvas_y):
    canvas_x = max(0.0, min(float(CANVAS_WIDTH), float(canvas_x)))
    canvas_y = max(0.0, min(float(CANVAS_HEIGHT), float(canvas_y)))

    robot_x = ROBOT_X_MIN + (canvas_y / CANVAS_HEIGHT) * (
        ROBOT_X_MAX - ROBOT_X_MIN
    )
    robot_y = ROBOT_Y_MIN + (canvas_x / CANVAS_WIDTH) * (
        ROBOT_Y_MAX - ROBOT_Y_MIN
    )

    robot_x = max(ROBOT_X_MIN, min(ROBOT_X_MAX, robot_x))
    robot_y = max(ROBOT_Y_MIN, min(ROBOT_Y_MAX, robot_y))
    return robot_x, robot_y


def robot_to_canvas(robot_x, robot_y):
    robot_x = max(ROBOT_X_MIN, min(ROBOT_X_MAX, float(robot_x)))
    robot_y = max(ROBOT_Y_MIN, min(ROBOT_Y_MAX, float(robot_y)))

    canvas_y = (
        (robot_x - ROBOT_X_MIN)
        / (ROBOT_X_MAX - ROBOT_X_MIN)
        * CANVAS_HEIGHT
    )
    canvas_x = (
        (robot_y - ROBOT_Y_MIN)
        / (ROBOT_Y_MAX - ROBOT_Y_MIN)
        * CANVAS_WIDTH
    )
    return canvas_x, canvas_y
