# Fall Detection System

A real-time fall detection system using computer vision and pose estimation to detect and alert when a person falls.

## Features

- Real-time fall detection using MediaPipe Pose
- Multiple detection metrics:
  - Vertical velocity
  - Torso angle
  - Position tracking
  - Height/width ratio analysis
- Recovery detection
- Alert system with:
  - Fall alerts with captured frames
  - Recovery alerts
  - Telegram integration
- Debug visualization with:
  - Pose landmarks
  - Bounding box
  - Real-time metrics display
  - Confidence scores

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- python-telegram-bot
- python-dotenv

## Telegram Bot Setup

1. Create a new Telegram bot:
   - Open Telegram and search for "@BotFather"
   - Start a chat with BotFather
   - Send `/newbot` command
   - Follow the prompts to:
     - Choose a name for your bot
     - Choose a username (must end in 'bot')
   - BotFather will give you a token (keep this secure)

2. Get your Chat ID:
   - Start a chat with your new bot
   - Send any message to the bot
   - Access this URL in your browser (replace with your bot token):
     ```
     https://api.telegram.org/bot<YourBOTToken>/getUpdates
     ```
   - Look for the "chat" object in the response, which contains your "id"
   - The response will look something like:
     ```json
     {
       "ok": true,
       "result": [{
         "message": {
           "chat": {
             "id": 123456789,  // This is your chat ID
             "first_name": "Your Name",
             "type": "private"
           }
         }
       }]
     }
     ```

3. Configure the environment:
   - Create a `.env` file in the project root
   - Add your bot token and chat ID:
     ```
     TELEGRAM_BOT_TOKEN=your_bot_token_here
     TELEGRAM_CHAT_ID=your_chat_id_here
     ```

4. Test the bot:
   - Run the application
   - The system will send a test message when started
   - You should receive alerts when falls are detected
   - You should receive recovery notifications when the person gets up

Note: Keep your bot token secure and never share it publicly. If your token is compromised, you can generate a new one using BotFather's `/revoke` command.

## Installation

1. Clone the repository:
```bash
git clone (https://github.com/olgavasilivna/fall-detection-alerts.git)
cd fall-detection-alerts
```

2. Create and activate a virtual environment:
```bash
python -m venv fall-app
source fall-app/bin/activate  # On Windows: fall-app\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Follow the Telegram Bot Setup instructions above to configure your `.env` file

## Usage

1. Run the system:
```bash
python main.py
```

2. The system will:
   - Open your webcam
   - Start detecting poses
   - Display real-time visualization
   - Send alerts when falls are detected
   - Send recovery alerts when person gets up

3. Press 'ctrl c' to quit the application

## Configuration

The system can be configured through the `FallDetector` class initialization:

```python
fall_detector = FallDetector({
    'min_confidence': 0.5,
    'min_consecutive_frames': 1,
    'window_size': 5,
    'base_velocity_threshold': 0.015,
    'base_torso_angle': 45,
    'debug': True
})
```

Key parameters:
- `min_confidence`: Minimum confidence threshold for pose detection
- `min_consecutive_frames`: Number of consecutive frames required for fall detection
- `window_size`: Size of the sliding window for smoothing
- `base_velocity_threshold`: Threshold for vertical velocity
- `base_torso_angle`: Threshold for torso angle
- `debug`: Enable/disable debug visualization

## Alert System

The system uses Telegram for alerts:
- Fall alerts include:
  - Captured frame of the fall
  - Confidence score
  - Timestamp
- Recovery alerts include:
  - Recovery confirmation
  - Time since fall
