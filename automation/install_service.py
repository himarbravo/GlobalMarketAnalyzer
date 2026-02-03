import os
import sys

# Configuración
LABEL = "com.himarbravoperez.globalmarketanalyst"
SCRIPT_NAME = "run_daily.sh"
TIME_HOUR = 8
TIME_MINUTE = 30

def create_plist(work_dir):
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{os.path.join(work_dir, SCRIPT_NAME)}</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{work_dir}</string>
    
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{TIME_HOUR}</integer>
        <key>Minute</key>
        <integer>{TIME_MINUTE}</integer>
    </dict>
    
    <key>StandardOutPath</key>
    <string>{os.path.join(work_dir, "logs", "launchd.out")}</string>
    
    <key>StandardErrorPath</key>
    <string>{os.path.join(work_dir, "logs", "launchd.err")}</string>
    
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""
    return plist_content

def install_service():
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Subir un nivel desde /automation
    plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist")
    
    print(f"📍 Directorio de trabajo detectado: {work_dir}")
    print(f"📄 Creando archivo plist en: {plist_path}")
    
    with open(plist_path, "w") as f:
        f.write(create_plist(work_dir))
        
    print("✅ Archivo creado.")
    
    # Cargar servicio
    print("🚀 Registrando servicio en launchd...")
    os.system(f"launchctl unload {plist_path} 2>/dev/null") # Limpiar previo si existe
    os.system(f"launchctl load {plist_path}")
    
    print("\n✨ ¡Instalación Completada!")
    print(f"El analista se ejecutará automáticamente todos los días a las {TIME_HOUR}:{TIME_MINUTE:02d}.")
    print("Para probarlo ahora mismo, ejecuta:")
    print(f"launchctl start {LABEL}")

if __name__ == "__main__":
    install_service()
