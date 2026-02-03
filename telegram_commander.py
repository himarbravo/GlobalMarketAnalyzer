import requests
import time
import subprocess
import config
import sys
import os

class TelegramCommander:
    def __init__(self):
        self.token = config.TELEGRAM_CONFIG["BOT_TOKEN"]
        self.allowed_id = str(config.TELEGRAM_CONFIG["CHAT_ID"])
        self.offset = 0
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        
        # Ensure we are in the project dir
        self.project_dir = os.path.dirname(os.path.abspath(__file__))

    def get_updates(self):
        try:
            url = f"{self.base_url}/getUpdates"
            params = {"offset": self.offset, "timeout": 30}
            response = requests.get(url, params=params, timeout=40)
            return response.json()
        except Exception as e:
            print(f"⚠️ Connection error: {e}")
            return {}

    def send_message(self, text):
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.allowed_id, "text": text}
            requests.post(url, json=payload)
        except:
            pass

    def run_analysis(self):
        self.send_message("🕵️‍♂️ Recibido. Iniciando análisis... esto tomará unos segundos.")
        
        try:
            # Run unified_system.py (The new brain)
            # Usa el python del venv actual
            python_exec = sys.executable
            script_path = os.path.join(self.project_dir, "unified_system.py")
            
            # Ejecutar y capturar salida
            result = subprocess.run([python_exec, script_path], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.send_message("✅ Análisis completado. El informe ha sido enviado.")
            else:
                self.send_message(f"❌ Error ejecutando análisis.\nLog:\n{result.stderr[-200:]}")
                
        except Exception as e:
            self.send_message(f"❌ Fallo crítico: {e}")

    def handle_command(self, command):
        cmd = command.lower().strip()
        
        if cmd == "/start":
            self.send_message("👋 Hola. Comandos disponibles:\n/analyze - Ejecutar análisis ahora\n/ping - Verificar estado")
        elif cmd == "/analyze":
            self.run_analysis()
        elif cmd == "/ping":
            self.send_message("🏓 Pong! El sistema está ONLINE y escuchando.")
        else:
            self.send_message("❓ Comando no reconocido. Prueba /analyze")

    def run_forever(self):
        print(f"🤖 Telegram Commander escuchando a usuario {self.allowed_id}...")
        self.send_message("🟢 Sistema de Control Remoto ACTIVADO.")
        
        while True:
            updates = self.get_updates()
            
            if updates.get("ok"):
                for update in updates.get("result", []):
                    update_id = update["update_id"]
                    self.offset = update_id + 1
                    
                    message = update.get("message", {})
                    chat_id = str(message.get("chat", {}).get("id"))
                    text = message.get("text", "")
                    
                    # Security Check
                    if chat_id != self.allowed_id:
                        print(f"⛔ Intento de acceso no autorizado de {chat_id}")
                        continue
                        
                    if text.startswith("/"):
                        print(f"📩 Comando recibido: {text}")
                        self.handle_command(text)
            
            time.sleep(1)

if __name__ == "__main__":
    bot = TelegramCommander()
    bot.run_forever()
