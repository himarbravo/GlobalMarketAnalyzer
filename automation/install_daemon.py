import os
import sys

# Configuración
LABEL = "com.himarbravoperez.globalmarketsystem"
SCRIPT_NAME = "start_system.sh"

def create_plist(work_dir):
    script_path = os.path.join(work_dir, SCRIPT_NAME)
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    
    <!-- Comando a ejecutar -->
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>{script_path}</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
    
    <!-- Ejecutar al iniciar sesión y mantener vivo (KeepAlive) -->
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    
    <!-- Logs -->
    <key>StandardOutPath</key>
    <string>{os.path.join(work_dir, "logs", "system_out.log")}</string>
    
    <key>StandardErrorPath</key>
    <string>{os.path.join(work_dir, "logs", "system_err.log")}</string>
</dict>
</plist>
"""
    return plist_content

def install_service():
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist")
    
    print(f"⚙️ Creando servicio 'Always On' para: {work_dir}")
    
    # 1. Crear logs dir si no existe
    os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)
    
    # 2. Escribir plist
    with open(plist_path, "w") as f:
        f.write(create_plist(work_dir))
        
    print(f"📄 Archivo PLIST generado: {plist_path}")
    
    # 3. Cagar servicio
    print("🚀 Registrando en macOS...")
    os.system(f"launchctl unload {plist_path} 2>/dev/null")
    os.system(f"launchctl load {plist_path}")
    
    print("\n✅ ¡Sistema Instalado en Segundo Plano!")
    print("El Bridge, la Dashboard y el Telegram Bot ahora funcionan invisibles.")
    print("Puedes cerrar la terminal sin miedo. El sistema se iniciará solo cuando enciendas el Mac.")
    print("Para detenerlo algún día: launchctl unload ~/Library/LaunchAgents/" + LABEL + ".plist")

if __name__ == "__main__":
    install_service()
