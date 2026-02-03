import requests
import time
import config

def setup():
    print("📱 Configuración de Alertas Telegram")
    print("===================================")
    print("Telegram no usa números de teléfono para los bots, usa 'Chat IDs'.")
    print("Sigue estos pasos rápidos:\n")
    
    # 1. Bot Token
    print("1. Abre Telegram y busca a '@BotFather'.")
    print("2. Escribe '/newbot' y dale un nombre cualquiera.")
    print("3. Te dará un TOKEN (largo, con letras y números).")
    
    token = input("\nPegar el TOKEN aquí: ").strip()
    
    if not token:
        print("❌ Necesito el token para continuar.")
        return

    # 2. Chat ID Discovery
    print(f"\n✅ Token recibido. Ahora conecta con tu bot.")
    print("1. Busca tu nuevo bot en Telegram (por el nombre que le diste).")
    print("2. Pulsa 'Iniciar' o escribe '/start'.")
    print("3. Vuelve aquí y pulsa ENTER.")
    input("\n(Pulsa ENTER cuando hayas iniciado tu bot)...")
    
    print("🔍 Buscando tu ID...")
    
    # Intentar obtener el ID via getUpdates (con reintentos)
    max_retries = 5
    for i in range(max_retries):
        print(f"   Intento {i+1}/{max_retries}...")
        try:
            url = f"https://api.telegram.org/bot{token}/getUpdates"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not data.get('ok'):
                if data.get('error_code') == 401:
                    print("❌ Error 401: Token inválido. Revisa que lo hayas copiado bien.")
                    return
                print(f"⚠️ Error Telegram: {data.get('description')}")
            
            if not data.get('result'):
                if i == 0:
                    print("⚠️ Aún no veo mensajes. ¿Le enviaste un 'Hola' al bot?")
                time.sleep(3)
                continue
                
            # Extraer Chat ID del último mensaje
            last_msg = data['result'][-1]
            chat_id = last_msg['message']['chat']['id']
            username = last_msg['message']['chat'].get('username', 'Usuario')
            
            print(f"\n🎉 ¡Te encontré, {username}!")
            print(f"Tu Chat ID es: {chat_id}")
            
            # 3. Guardar en config.py
            save = input("\n¿Quieres guardar esto en config.py ahora? (s/n): ").lower()
            if save == 's' or save == '':
                update_config(token, str(chat_id))
            return

        except Exception as e:
            print(f"❌ Error de red: {e}")
            time.sleep(2)
            
    print("\n😪 No logré detectar mensajes después de varios intentos.")
    print("Consejo: Envía otro mensaje 'Hola' a tu bot y vuelve a ejecutar este script.")

def update_config(token, chat_id):
    # Leer archivo actual
    with open("config.py", "r") as f:
        lines = f.readlines()
    
    # Reescribir con los nuevos valores
    new_lines = []
    in_telegram_block = False
    
    for line in lines:
        if "TELEGRAM_CONFIG = {" in line:
            in_telegram_block = True
            new_lines.append(line)
            new_lines.append(f'    "ENABLED": True,\n')
            new_lines.append(f'    "BOT_TOKEN": "{token}",\n')
            new_lines.append(f'    "CHAT_ID": "{chat_id}"\n')
            new_lines.append("}\n")
        elif in_telegram_block and "}" in line:
            in_telegram_block = False # Fin del bloque original
        elif in_telegram_block:
            continue # Saltar las líneas viejas del bloque
        else:
            new_lines.append(line)
            
    with open("config.py", "w") as f:
        f.writelines(new_lines)
        
    print("\n✅ ¡Configuración guardada! Tus alertas están activas.")
    print("Prueba ejecutando: .venv/bin/python market_analyst.py")

if __name__ == "__main__":
    setup()
