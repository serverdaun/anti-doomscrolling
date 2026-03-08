import logging
import time

import cv2

from src.core.config import VisionConfig
from src.core.logging import setup_logging
from src.core.state_machine import DoomscrollStateMachine
from src.notifications import MacOSNotifier, NotificationDispatcher
from src.vision.capture.camera import Camera
from src.vision.overlay import Overlay
from src.vision.pipeline import VisionPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    config = VisionConfig()
    logger.info(
        "Config: fps=%d, yolo_every_n=%d, device=%s",
        config.target_fps,
        config.yolo_every_n,
        config.yolo_device,
    )

    camera = Camera(config)
    pipeline = VisionPipeline(config)
    state_machine = DoomscrollStateMachine(config)
    overlay = Overlay()

    dispatcher = NotificationDispatcher()
    if config.notifications_enabled:
        dispatcher.add(MacOSNotifier(cooldown=config.notification_cooldown))

    frame_interval = 1.0 / config.target_fps
    fps = 0.0
    prev_state = state_machine.state

    camera.start()
    logger.info("Anti-doomscroll detector running. Press 'q' to quit.")

    try:
        while True:
            t_start = time.perf_counter()

            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            result = pipeline.process(frame)

            phone_detected = len(result.phones) > 0
            looking_at_screen = result.gaze.looking_at_screen if result.gaze else False
            face_detected = result.face is not None

            state = state_machine.update(
                phone_detected, looking_at_screen, face_detected
            )

            if state != prev_state:
                dispatcher.on_state_change(state, prev_state)
                prev_state = state

            annotated = overlay.draw(
                frame,
                result.face,
                result.gaze,
                result.phones,
                state,
                fps,
                result.process_ms,
            )

            cv2.imshow("Anti-Doomscroll Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Throttle to target FPS
            elapsed = time.perf_counter() - t_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            total = time.perf_counter() - t_start
            fps = 1.0 / total if total > 0 else 0.0

    finally:
        logger.info("Shutting down")
        camera.stop()
        pipeline.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
