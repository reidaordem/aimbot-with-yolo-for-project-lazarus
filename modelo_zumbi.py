# modelo_zumbi_tela_cheia.py
import cv2
import numpy as np
import mss
import win32api
import win32con
import time
import onnxruntime as ort

# ---------- CONFIGURAÇÕES ----------
MODEL_PATH = r"C:\Users\Usuario\Desktop\meurobopython\runs\detect\train3\weights\best.onnx"
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

# Centro da tela inteira
center_x = monitor["width"] / 2.0
center_y = monitor["height"] / 2.0

scale_x = 960 / 1920
scale_y = 540 / 1080

# Sensibilidade do mouse - AJUSTE ESTES VALORES!
COUNTS_PER_PIXEL_X = 1.0 / 2.0  # Reduzido para movimento mais suave
COUNTS_PER_PIXEL_Y = 1.0 / 2.0

# Filtros
CONF_TH = 0.6
MIN_BOX_AREA = 1000  # Aumentado para tela maior

# Otimizações de movimento
SMOOTHING = False
SMOOTHING_FACTOR = 0.3
DEADZONE = 20  # Área onde não move o mouse

# inicializa ONNX
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

sct = mss.mss()

print("Usando tela inteira:", monitor["width"], "x", monitor["height"])
print("Centro da tela:", center_x, center_y)

def pixel_to_mouse_move(px, py):
    """
    Converte coordenadas absolutas da tela para movimento relativo do mouse
    """
    dx_pix = px - center_x
    dy_pix = py - center_y
    
    # Deadzone - não move se estiver muito perto do centro
    if abs(dx_pix) < DEADZONE and abs(dy_pix) < DEADZONE:
        return 0, 0
    
    # Movimento não-linear - mais preciso perto do centro, mais rápido longe
    distance = np.sqrt(dx_pix**2 + dy_pix**2)
    scale_factor = min(2.0, 1.0 + distance / 500.0)  # Limita o fator de escala
    
    mouse_dx = dx_pix * COUNTS_PER_PIXEL_X * scale_factor
    mouse_dy = -dy_pix * COUNTS_PER_PIXEL_Y * scale_factor
    
    return mouse_dx, mouse_dy

def process_yolo_output(outputs, conf_threshold=0.5):
    """Processamento da saída YOLO para tela inteira"""
    predictions = np.squeeze(outputs[0]).T
    
    boxes = []
    for pred in predictions:
        x_center, y_center, width, height, confidence = pred
        
        if confidence < conf_threshold:
            continue
            
        # Converter coordenadas normalizadas para pixels na tela inteira
        x1 = int((x_center - width/2) * monitor["width"] / 640)
        y1 = int((y_center - height/2) * monitor["height"] / 640)
        x2 = int((x_center + width/2) * monitor["width"] / 640)
        y2 = int((y_center + height/2) * monitor["height"] / 640)
        
        # Clamping
        x1 = max(0, min(monitor["width"]-1, x1))
        y1 = max(0, min(monitor["height"]-1, y1))
        x2 = max(0, min(monitor["width"]-1, x2))
        y2 = max(0, min(monitor["height"]-1, y2))
        
        # Filtro de área
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            continue
            
        boxes.append((x1, y1, x2, y2, confidence))
    
    return boxes

# Variáveis para suavização
last_move_x, last_move_y = 0, 0
frame_count = 0

try:
    print("Iniciando detecção em tela cheia...")
    print("Pressione 'q' para sair")
    
    while True:
        t0 = time.time()
        
        # Captura tela inteira
        sct_img = sct.grab(monitor)
        frame = np.frombuffer(sct_img.rgb, dtype=np.uint8).reshape((sct_img.height, sct_img.width, 3))
        
        # Reduz tamanho para debug (opcional)
        debug_frame = cv2.resize(frame, (960, 540))  # 50% do original

        # Pré-processamento para o modelo
        img = cv2.resize(frame, (640, 640))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # Inferência
        outputs = sess.run([output_name], {input_name: img_tensor})[0]
        
        # Processamento
        detections = process_yolo_output([outputs], CONF_TH)
        
        # Escolher melhor detecção
        best = None
        if detections:
            best = max(detections, key=lambda b: b[4])

        if best is not None:
            bx1, by1, bx2, by2, bconf = best
            px = (bx1 + bx2) / 2.0
            py = (by1 + by2) / 2.0

            # Calcular movimento do mouse
            raw_dx, raw_dy = pixel_to_mouse_move(px, py)
            
            # Suavização
            if SMOOTHING:
                smooth_dx = SMOOTHING_FACTOR * raw_dx + (1 - SMOOTHING_FACTOR) * last_move_x
                smooth_dy = SMOOTHING_FACTOR * raw_dy + (1 - SMOOTHING_FACTOR) * last_move_y
                last_move_x, last_move_y = smooth_dx, smooth_dy
                nX, nY = int(round(smooth_dx)), int(round(smooth_dy))
            else:
                nX, nY = int(round(raw_dx)), int(round(raw_dy))

            # Aplicar movimento
            if nX != 0 or nY != 0:
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, nX, nY, 0, 0)

            # Debug visual no frame reduzido
            
            
            dbx1 = int(bx1 * scale_x)
            dby1 = int(by1 * scale_y)
            dbx2 = int(bx2 * scale_x)
            dby2 = int(by2 * scale_y)
            dpx = int(px * scale_x)
            dpy = int(py * scale_y)
            dcenter_x = int(center_x * scale_x)
            dcenter_y = int(center_y * scale_y)
            
            cv2.rectangle(debug_frame, (dbx1, dby1), (dbx2, dby2), (0, 255, 0), 2)
            cv2.circle(debug_frame, (dpx, dpy), 4, (0, 0, 255), -1)
            cv2.line(debug_frame, (dcenter_x, dcenter_y), (dpx, dpy), (255, 0, 0), 2)
            
            # Informações
            cv2.putText(debug_frame, f"CONF: {bconf:.3f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"MOV: ({nX}, {nY})", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if frame_count % 10 == 0:
                print(f"DETECÇÃO: conf={bconf:.3f}, centro=({px:.0f},{py:.0f}), movimento=({nX},{nY})")

        # Centro e FPS
        debug_center_x = int(center_x * scale_x)
        debug_center_y = int(center_y * scale_y)
        cv2.circle(debug_frame, (debug_center_x, debug_center_y), 3, (255, 255, 0), -1)
        
        fps = 1.0 / (time.time() - t0)
        cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Debug - Tela Cheia (50%)", debug_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

finally:
    cv2.destroyAllWindows()