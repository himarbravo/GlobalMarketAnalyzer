
import sys
import os
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

# --- CONFIGURACIÓN DE RUTAS ---
# Ajustamos para poder importar 'lab' desde el proyecto hermano
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_PROJECT_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "../kernel_embeding_p"))
sys.path.append(KERNEL_PROJECT_PATH)

print(f"🌉 Bridge Server iniciando...")
print(f"📂 Importando motor desde: {KERNEL_PROJECT_PATH}")

try:
    from lab.engine.generator import LLMEngine
    print("✅ Motor importado correctamente.")
except ImportError as e:
    print(f"❌ Error importando LLMEngine: {e}")
    print("Asegúrate de ejecutar este script con el Python del entorno virtual de 'kernel_embeding_p'.")
    sys.exit(1)

# --- INICIALIZAR MOTOR ---
# Usamos un modelo ligero por defecto o el que definan
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # De los options vistos en main.py
print(f"🚀 Cargando modelo: {MODEL_NAME}...")
try:
    engine = LLMEngine(MODEL_NAME)
    print(f"✨ Modelo cargado en {engine.device}")
except Exception as e:
    print(f"❌ Error fatal cargando modelo: {e}")
    sys.exit(1)

# --- SERVIDOR HTTP ---
class AIRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                prompt = data.get('prompt', '')
                
                # Parsear formato Ollama (System + User) si viene así
                # El script market_analyst envía: "System: ... \nUser: ..."
                # El motor LLMEngine soporta lista de mensajes o texto raw.
                # Vamos a intentar convertirlo a lista de chats si detectamos estructura
                
                messages = []
                if "System:" in prompt and "User:" in prompt:
                    parts = prompt.split("User:")
                    sys_msg = parts[0].replace("System:", "").strip()
                    user_msg = parts[1].strip()
                    messages = [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ]
                    input_payload = messages
                else:
                    input_payload = prompt

                # Generar
                print(f"📨 Procesando solicitud ({len(prompt)} chars)...")
                response_text = engine.generate(
                    input_payload,
                    max_new_tokens=300,
                    temperature=0.7
                )
                
                # Responder formato JSON estilo Ollama
                response_data = {"response": response_text}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                print(f"🔥 Error en inferencia: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"error": "Internal Server Error"}')
        else:
            self.send_response(404)
            self.end_headers()

def run(server_class=HTTPServer, handler_class=AIRequestHandler, port=5050):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"🟢 Servidor AI Bridge escuchando en puerto {port}...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
