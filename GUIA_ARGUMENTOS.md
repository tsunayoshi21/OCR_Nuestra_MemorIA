# 📖 Guía de Argumentos de Línea de Comandos

## 🎯 Uso Básico

```bash
# Ejecutar con configuración por defecto (del diccionario CONFIG)
python olmocr_run_better_fallback.py

# Ver ayuda completa
python olmocr_run_better_fallback.py --help
```

## 📋 Argumentos Disponibles

### 📁 **Procesamiento de PDFs**

```bash
--pdf-folder PATH
```
Ruta a la carpeta que contiene los PDFs a procesar.

**Ejemplo:**
```bash
python olmocr_run_better_fallback.py --pdf-folder /ruta/a/mis/pdfs
```

---

### 🤖 **Configuración de Modelos OCR**

```bash
--reco-model MODEL_NAME
```
Modelo de reconocimiento OCR a utilizar.

**Opciones:**
- `DHiSS_v1_parseq`
- `DHiSS_v1_sar_resnet31`
- `DHiSS_v1_vitstr_base`
- `DHiSS_v2_vitstr_base`
- `DHiSS_v2_parseq`

**Ejemplo:**
```bash
python olmocr_run_better_fallback.py --reco-model DHiSS_v2_parseq
```

```bash
--detector-model MODEL_NAME
```
Modelo detector a utilizar (default: `db_resnet50`).

---

### 💾 **Configuración de Salida**

```bash
--output-folder NAME
```
Nombre de la carpeta donde se guardarán los resultados.

**Ejemplo:**
```bash
python olmocr_run_better_fallback.py --output-folder mis_resultados_OCR
```

```bash
--log-file FILENAME
```
Nombre del archivo de log (default: `output_folder_name.log`).

**Ejemplo:**
```bash
python olmocr_run_better_fallback.py --log-file experimento_1.log
```

---

### 🔧 **OCR Personalizado**

```bash
--custom-ocr          # Habilitar OCR personalizado
--no-custom-ocr       # Deshabilitar OCR personalizado (usar estándar)
```

**Ejemplo:**
```bash
# Habilitar OCR personalizado
python olmocr_run_better_fallback.py --custom-ocr --reco-model DHiSS_v2_vitstr_base

# Deshabilitar (más rápido, menos preciso)
python olmocr_run_better_fallback.py --no-custom-ocr
```

---

### 🌡️ **Estrategia de Temperatura (Fallback)**

```bash
--temperatures T1 T2 T3 ...
```
Lista de temperaturas a probar en orden (valores entre 0 y 2).

**Ejemplo:**
```bash
# Más conservador
python olmocr_run_better_fallback.py --temperatures 0.05 0.2 0.5

# Más creativo
python olmocr_run_better_fallback.py --temperatures 0.3 0.7 1.2
```

```bash
--final-temperature VALUE
```
Temperatura para el fallback final con prompt vacío.

**Ejemplo:**
```bash
python olmocr_run_better_fallback.py --final-temperature 0.05
```

---

### 🧠 **Parámetros del Modelo**

```bash
--max-tokens NUMBER
```
Número máximo de tokens nuevos a generar.

**Ejemplo:**
```bash
python olmocr_run_better_fallback.py --max-tokens 8192  # Para documentos muy largos
```

```bash
--num-sequences NUMBER
```
Número de secuencias a retornar por generación.

---

### 🖼️ **Resolución de Imagen**

```bash
--target-dim PIXELS
```
Dimensión objetivo del lado más largo de la imagen (en píxeles).

**Valores comunes:**
- `1024` - Estándar, rápido
- `1536` - Media-alta resolución
- `2048` - Alta resolución
- `4096` - Muy alta resolución (requiere GPU potente)

**Ejemplo:**
```bash
python olmocr_run_better_fallback.py --target-dim 2048
```

```bash
--min-pixels NUMBER
```
Número mínimo de píxeles para procesamiento (default: `200,704` ≈ 256×28×28).

```bash
--max-pixels NUMBER
```
Número máximo de píxeles para procesamiento.

**Cálculo:**
```
max_pixels = target_dim × 28 × 28
```

**Ejemplos:**
```bash
# 1024px
python olmocr_run_better_fallback.py --max-pixels 802816

# 2048px
python olmocr_run_better_fallback.py --max-pixels 1605632

# 4096px
python olmocr_run_better_fallback.py --max-pixels 4063232
```

---

### 🎯 **Parámetros de OCR**

```bash
--ocr-threshold VALUE
```
Umbral de confianza para filtrar palabras OCR (0.0 a 1.0).

**Valores:**
- `0.85` - Menos selectivo, más palabras
- `0.91` - Balance (default)
- `0.95` - Muy selectivo, solo palabras de alta confianza

**Ejemplo:**
```bash
python olmocr_run_better_fallback.py --ocr-threshold 0.85
```

---

### 🎮 **Configuración de GPU**

```bash
--gpu {0,1,2}
```
ID del dispositivo GPU a utilizar para el procesamiento.

**Opciones:**
- `0` - GPU 0 (default)
- `1` - GPU 1
- `2` - GPU 2

**Ejemplo:**
```bash
# Usar GPU 1
python olmocr_run_better_fallback.py --gpu 1

# Usar GPU 2 con configuración específica
python olmocr_run_better_fallback.py --gpu 2 --target-dim 4096
```

**Nota:** Asegúrate de tener la GPU disponible antes de especificar su ID. Verifica con `nvidia-smi`.

---

### 📄 **Formato de Salida**

```bash
--page-separator          # Agregar separadores de página
--no-page-separator       # No agregar separadores (default)
```
Añade marcadores de página en el texto extraído (ej: "Page (1)", "Page (2)").

**Ejemplo:**
```bash
# Con separadores de página
python olmocr_run_better_fallback.py --page-separator

# Sin separadores (comportamiento por defecto)
python olmocr_run_better_fallback.py --no-page-separator
```

**Salida con `--page-separator`:**
```
Page (1)
[Texto de la página 1]

Page (2)
[Texto de la página 2]
```

**Salida sin separador (default):**
```
[Texto de la página 1]

[Texto de la página 2]
```

---

## 🚀 Ejemplos de Uso Completo

### Ejemplo 1: Alta Resolución con Custom OCR
```bash
python olmocr_run_better_fallback.py \
    --pdf-folder /home/user/documentos \
    --output-folder experimento_alta_res \
    --target-dim 2048 \
    --max-pixels 1605632 \
    --custom-ocr \
    --reco-model DHiSS_v2_vitstr_base \
    --temperatures 0.1 0.4 0.8 \
    --ocr-threshold 0.85 \
    --gpu 0 \
    --page-separator
```

### Ejemplo 2: Procesamiento Rápido (Baja Resolución)
```bash
python olmocr_run_better_fallback.py \
    --pdf-folder /home/user/documentos \
    --output-folder rapido_1024 \
    --target-dim 1024 \
    --max-pixels 802816 \
    --no-custom-ocr \
    --temperatures 0.2 0.5 \
    --max-tokens 2048 \
    --gpu 0
```

### Ejemplo 3: Máxima Calidad (GPU Potente)
```bash
python olmocr_run_better_fallback.py \
    --pdf-folder /home/user/documentos \
    --output-folder maxima_calidad \
    --target-dim 4096 \
    --max-pixels 4063232 \
    --custom-ocr \
    --reco-model DHiSS_v2_parseq \
    --temperatures 0.05 0.2 0.5 0.9 \
    --final-temperature 0.05 \
    --max-tokens 8192 \
    --ocr-threshold 0.95 \
    --gpu 1 \
    --page-separator
```

### Ejemplo 4: Procesamiento Paralelo en Múltiples GPUs
```bash
# GPU 0 - Resolución 1024px
python olmocr_run_better_fallback.py \
    --pdf-folder /home/user/documentos \
    --output-folder gpu0_1024px \
    --target-dim 1024 \
    --gpu 0 \
    --page-separator &

# GPU 1 - Resolución 2048px
python olmocr_run_better_fallback.py \
    --pdf-folder /home/user/documentos \
    --output-folder gpu1_2048px \
    --target-dim 2048 \
    --gpu 1 \
    --page-separator &

# GPU 2 - Resolución 4096px
python olmocr_run_better_fallback.py \
    --pdf-folder /home/user/documentos \
    --output-folder gpu2_4096px \
    --target-dim 4096 \
    --gpu 2 \
    --page-separator &

wait
echo "Procesamiento paralelo completado"
```

---

## 🔁 Scripts Bash para Múltiples Configuraciones

### Script Simple: Probar 3 Resoluciones
```bash
#!/bin/bash
PDF_FOLDER="/ruta/a/pdfs"

for DIM in 1024 1536 2048; do
    python olmocr_run_better_fallback.py \
        --pdf-folder "$PDF_FOLDER" \
        --output-folder "olmOCR_${DIM}px" \
        --target-dim $DIM \
        --max-pixels $((DIM * 28 * 28)) \
        --no-custom-ocr \
        --page-separator
done
```

### Script Avanzado: Comparar Modelos
```bash
#!/bin/bash
PDF_FOLDER="/ruta/a/pdfs"
MODELS=("DHiSS_v2_parseq" "DHiSS_v2_vitstr_base" "DHiSS_v1_vitstr_base")

for MODEL in "${MODELS[@]}"; do
    python olmocr_run_better_fallback.py \
        --pdf-folder "$PDF_FOLDER" \
        --output-folder "olmOCR_${MODEL}" \
        --reco-model "$MODEL" \
        --custom-ocr \
        --target-dim 2048 \
        --max-pixels 1605632 \
        --page-separator
done
```

### Script Completo: Grid Search
```bash
#!/bin/bash
PDF_FOLDER="/ruta/a/pdfs"

# Resoluciones a probar
RESOLUTIONS=(1024 2048)

# Modelos a probar
MODELS=("DHiSS_v2_parseq" "DHiSS_v2_vitstr_base")

# Iterar sobre todas las combinaciones
for DIM in "${RESOLUTIONS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        OUTPUT_NAME="olmOCR_${MODEL}_${DIM}px"
        
        echo "Procesando: $OUTPUT_NAME"
        
        python olmocr_run_better_fallback.py \
            --pdf-folder "$PDF_FOLDER" \
            --output-folder "$OUTPUT_NAME" \
            --reco-model "$MODEL" \
            --target-dim $DIM \
            --max-pixels $((DIM * 28 * 28)) \
            --custom-ocr \
            --temperatures 0.1 0.4 0.8
        
        echo "Completado: $OUTPUT_NAME"
        echo "---"
    done
done
```

---

## 📊 Tabla de Resoluciones Recomendadas

| target_dim | max_pixels | VRAM (aprox) | Uso Recomendado |
|------------|------------|--------------|-----------------|
| 1024       | 802,816    | 8-12 GB      | Rápido, documentos simples |
| 1536       | 1,204,224  | 12-16 GB     | Balance calidad/velocidad |
| 2048       | 1,605,632  | 16-24 GB     | Alta calidad, texto pequeño |
| 4096       | 4,063,232  | 32-48 GB     | Máxima calidad, GPU potente |

---

## 💡 Tips y Trucos

### 1. **Ver Ayuda**
```bash
python olmocr_run_better_fallback.py --help
```

### 2. **Combinar con Default CONFIG**
Los argumentos de línea de comandos **sobrescriben** los valores del CONFIG. Si no pasas un argumento, se usa el valor del CONFIG.

```bash
# Cambia solo la carpeta, usa resto del CONFIG
python olmocr_run_better_fallback.py --pdf-folder /nueva/ruta
```

### 3. **Logs Organizados**
```bash
# Nombrar logs según experimento
python olmocr_run_better_fallback.py \
    --output-folder exp1_2048px \
    --log-file exp1_2048px.log
```

### 4. **Ejecución en Background**
```bash
# Ejecutar en background y guardar output
nohup python olmocr_run_better_fallback.py \
    --output-folder experimento_largo \
    --target-dim 4096 \
    > output.txt 2>&1 &
```

### 5. **Monitorear Progreso**
```bash
# En otra terminal
tail -f olmOCR_experimento.log
```

---

## ⚠️ Consideraciones Importantes

1. **VRAM**: Asegúrate de tener suficiente VRAM para la resolución elegida
2. **Disk Space**: Alta resolución = archivos de log más grandes
3. **Tiempo**: 4096px puede ser 4-8x más lento que 1024px
4. **Custom OCR**: Requiere que los archivos `.pt` estén en `ocr_weights/`

---

## 🆘 Troubleshooting

### Error: "CUDA out of memory"
**Solución:**
```bash
# Reducir resolución
python olmocr_run_better_fallback.py --target-dim 1024 --max-pixels 802816
```

### Error: "PDF folder does not exist"
**Solución:**
```bash
# Verificar ruta con path absoluto
python olmocr_run_better_fallback.py --pdf-folder /home/user/PDFs
```

### Error: "Model checkpoint not found"
**Solución:**
```bash
# Verificar que existe ocr_weights/DHiSS_finetuning_*.pt
ls ocr_weights/
```

---

¡Disfruta experimentando con diferentes configuraciones! 🚀
