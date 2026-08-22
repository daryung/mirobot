import pyrealsense2 as rs
import numpy as np
import cv2


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

clicked_x = -1
clicked_y = -1


def mouse_callback(event, x, y, flags, param):
    global clicked_x, clicked_y

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_x = x
        clicked_y = y


cv2.namedWindow("RealSense")
cv2.setMouseCallback("RealSense", mouse_callback)


try:
    while True:

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

        display_image = color_image.copy()

        if clicked_x >= 0 and clicked_y >= 0:

            depth = depth_frame.get_distance(clicked_x, clicked_y)

            if depth > 0:

                point = rs.rs2_deproject_pixel_to_point(
                    intrinsics,
                    [clicked_x, clicked_y],
                    depth
                )

                x_mm = point[0] * 1000
                y_mm = point[1] * 1000
                z_mm = point[2] * 1000

                print("\n==============================")
                print(f"Pixel : ({clicked_x}, {clicked_y})")
                print(f"Depth : {depth:.4f} m")
                print(f"Camera X : {x_mm:.2f} mm")
                print(f"Camera Y : {y_mm:.2f} mm")
                print(f"Camera Z : {z_mm:.2f} mm")
                print("==============================")

                cv2.circle(
                    display_image,
                    (clicked_x, clicked_y),
                    5,
                    (0, 0, 255),
                    -1
                )

                text = f"X:{x_mm:.1f} Y:{y_mm:.1f} Z:{z_mm:.1f} mm"

                cv2.putText(
                    display_image,
                    text,
                    (clicked_x + 10, clicked_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

            else:
                print("Depth 값을 얻을 수 없습니다.")

            clicked_x = -1
            clicked_y = -1

        cv2.imshow("RealSense", display_image)

        key = cv2.waitKey(1)

        if key == ord('q') or key == 27:
            break


finally:
    pipeline.stop()
    cv2.destroyAllWindows()