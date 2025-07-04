 Low-Cost AI-Based PPE Compliance Detection System


 Project Overview

This project presents a low-cost AI-based system that automatically
detects if workers are wearing Personal Protective Equipment (PPE)
like helmets and safety vests in industrial settings (e.g., construction
sites). It uses YOLO object detection, a Raspberry Pi 5, a
camera, and a GSM module to provide real-time alerts through
buzzer sound and SMS messages.
  

Group Members (ENG GROUP 4 -- UICT)

-   Mulindwa Mark -- 2023/DEE/DAY/1226/G
-   Kyasiimire Allen -- 2023/DEE/DAY/1235/G
-   Aturinda Prosca -- 2023/DEE/DAY/1695/G
-   Ayesiga Derrick -- 2023/DEE/DAY/0354
-   Ssegane Christopher -- 2023/DEE/DAY/1210/G

------------------------------------------------------------------------

  Features

-    Real-time helmet and vest detection using YOLOv11
-    Buzzer alert for immediate feedback to the worker
-    SMS alert to supervisor via SIM800L GSM module
-    Image logging for violations with timestamp
-    Low-cost and easy to set up using open-source tools

------------------------------------------------------------------------

  System Components

  Component            Description
  -------------------- ------------------------------
  Raspberry Pi 5       Edge device for AI inference
  Pi Camera V2         Captures real-time images
  YOLOv11 Model        Object detection algorithm
  SIM800L GSM Module   Sends SMS alerts
  Buzzer (85dB)        Audible warning
  SD Card (32GB)       Stores OS, logs, and images

------------------------------------------------------------------------

  How It Works

1.  Camera-captures live feed of workers.
2.  YOLO model-detects if helmet and safety vest are worn.
3.  If PPE is missing:
    -    Buzzer is activated
    -    SMS is sent via GSM module
    -    Image of violation is saved locally


 Sample Output

-   ![Helmet Violation](images/sample_violation.png)
    Detected: No Helmet --- SMS sent and buzzer triggered.

------------------------------------------------------------------------

 Prerequisites

-   Python 3.9
-   Raspberry Pi OS (Bookworm or Bullseye)
-   OpenCV 4.x
-   PyTorch
-   RPi.GPIO
-   pySerial


