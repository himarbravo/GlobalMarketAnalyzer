import requests
import config

class Notifier:
    def __init__(self):
        self.enabled = config.TELEGRAM_CONFIG["ENABLED"]
        self.token = config.TELEGRAM_CONFIG["BOT_TOKEN"]
        self.chat_id = config.TELEGRAM_CONFIG["CHAT_ID"]
        
    def send_alert(self, message):
        """Envía un mensaje a Telegram."""
        if not self.enabled or self.token == "TU_TOKEN_AQUI":
            print("🔕 Notificaciones desactivadas (Falta Token).")
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                print("📲 Notificación enviada al móvil.")
            else:
                print(f"⚠️ Error Telegram: {response.text}")
        except Exception as e:
            print(f"⚠️ Fallo de red Telegram: {e}")

    def format_morning_brief(self, market_narrative, top_picks, risk_alert):
        """Crea un mensaje bonito para despertar al usuario."""
        
        msg = f"☕ **Morning Brief**\n\n"
        msg += f"🌍 {market_narrative}\n\n"
        
        if top_picks:
            msg += "**🚀 Oportunidades (Top Picks):**\n"
            for pick in top_picks:
                msg += f"• **{pick['Ticker']}**: {pick['Signal']} (Conv: {pick['Confidence_Score']:.1f})\n"
        else:
            msg += "🧘 Día de paciencia. Sin entradas fuertes.\n"
            
        if "Riesgo" in risk_alert or "Duplicate" in risk_alert:
            msg += f"\n🚨 **Alerta de Cartera**:\n{risk_alert[:100]}..."
            
        return msg
