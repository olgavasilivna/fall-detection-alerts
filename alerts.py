import os
import time
from typing import Optional
from dotenv import load_dotenv
import requests

class AlertSystem:
    def __init__(self):
        """Initialize the alert system."""
        load_dotenv()
        self.last_alert_time = 0
        self.alert_cooldown = 0.0  # No cooldown to ensure immediate alerts
        
        # Get Telegram configuration
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.telegram_token or not self.telegram_chat_id:
            print("Warning: Telegram configuration not found. Alerts will only be printed to console.")

    def send_telegram_alert(self, message: str, image_path: Optional[str] = None) -> bool:
        """Send alert to Telegram."""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
            
        try:
            # Send message
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            # Send image if provided
            if image_path and os.path.exists(image_path):
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
                with open(image_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {"chat_id": self.telegram_chat_id}
                    response = requests.post(url, data=data, files=files)
                    response.raise_for_status()
            
            return True
        except Exception as e:
            print(f"Error sending Telegram alert: {e}")
            return False

    def send_alert_sync(self, confidence: float, frame_path: Optional[str] = None) -> bool:
        """
        Send an alert synchronously.
        
        Args:
            confidence: Confidence score of the fall detection
            frame_path: Optional path to the frame image
            
        Returns:
            bool: True if alert was sent successfully
        """
        # Create alert message
        message = (
            f"🚨 <b>FALL ALERT</b> 🚨\n\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Confidence: {confidence:.2f}\n"
        )
        
        # Print to console
        print("\n" + "!"*50)
        print("!"*20 + " FALL ALERT " + "!"*20)
        print("!"*50)
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Confidence: {confidence:.2f}")
        if frame_path:
            print(f"Frame saved: {frame_path}")
        print("!"*50 + "\n")
        
        # Send to Telegram
        telegram_success = self.send_telegram_alert(message, frame_path)
        
        # Force flush to ensure message appears immediately
        import sys
        sys.stdout.flush()
        
        return telegram_success 

    def send_recovery_alert(self, metrics: dict) -> bool:
        """Send a recovery alert to Telegram."""
        message = (
            f"✅ <b>RECOVERY ALERT</b> ✅\n\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Status: Person has recovered from fall\n"
        )
        return self.send_telegram_alert(message) 