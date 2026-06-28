      All original images are stored in a folder named original, the detection results of YOLOv12s are stored in a folder named base, and the detection results of HQ-YOLO are stored in a folder named our.
      It is worth noting that we use YOLOv12s and HQ-YOLO to detect the video named v1, which has a frame rate of 30 and a duration of 12 seconds. We used all consecutive frames (362 images) for the experiment. YOLOv12s detected traffic signs in 178 images, while HQ-YOLO detected traffic signs in 215 images.
v1: The original video.
v1_base: The video obtained through detection with YOLOv12s.
v1_our: The video obtained through detection with HQ-YOLO.
Note: Frame by frame detection is similar to the whole video detection in concept, but there may be differences in actual implementation and effect.