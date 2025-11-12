# OCR Nuestra MemorIA

Sistema de OCR para procesamiento de documentos PDF utilizando OlmOCR con modelos personalizados DHiSS. Este proyecto incluye un mecanismo de fallback basado en temperatura para una extracción de texto robusta.

## 📋 Tabla de Contenidos

- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Descarga de Modelos OCR](#-descarga-de-modelos-ocr)
- [Configuración](#️-configuración)
- [Uso](#-uso)
  - [Ejecución con Python](#ejecución-con-python)
  - [Ejecución con Scripts Bash](#ejecución-con-scripts-bash)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Logs y Resultados](#-logs-y-resultados)
- [Modelos Soportados](#-modelos-soportados)

## 🔧 Requisitos

- **Python**: 3.11
- **CUDA**: Compatible con GPU NVIDIA (recomendado para mejor rendimiento)
- **Conda**: Para gestión de ambientes
- **Espacio en disco**: 
  - ~10 GB para el ambiente conda y dependencias
  - ~5-10 GB adicionales para los pesos de los modelos OCR

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/OCR_Nuestra_MemorIA.git
cd OCR_Nuestra_MemorIA
```

### 2. Crear el ambiente Conda

```bash
# Crear el ambiente desde el archivo environment.yml
conda env create -f environment.yml

# Activar el ambiente
conda activate olmo_doc
```

### 3. Verificar la instalación

```bash
# Verificar que Python esté correctamente instalado
python --version  # Debe mostrar Python 3.11.x

# Verificar que PyTorch detecte la GPU
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

## 📥 Descarga de Modelos OCR

⚠️ **IMPORTANTE**: Los pesos de los modelos OCR son archivos grandes (~500MB - 2GB cada uno) y **NO están incluidos en el repositorio**. Debes descargarlos por separado.

### Ubicación de los modelos

Los modelos deben colocarse en la carpeta `ocr_weights/` en la raíz del proyecto:

```
OCR_Nuestra_MemorIA/
├── ocr_weights/
│   ├── DHiSS_finetuning_parseq_10.pt
│   ├── DHiSS_finetuning_v2_parseq_10.pt
│   ├── DHiSS_finetuning_v2_vitstr_base_10.pt
│   └── DHiSS_finetuning_vitstr_base_10.pt
├── ocr_inference.py
├── environment.yml
└── README.md
```

### Crear la carpeta de modelos

```bash
# Crear la carpeta si no existe
mkdir -p ocr_weights
```

### Descargar los modelos

Descarga los archivos `.pt` desde [ubicación de los modelos] y colócalos en `ocr_weights/`:

```bash
# Ejemplo (ajusta la URL según corresponda)
cd ocr_weights/
# wget <URL_DEL_MODELO>/DHiSS_finetuning_v2_vitstr_base_10.pt
# wget <URL_DEL_MODELO>/DHiSS_finetuning_v2_parseq_10.pt
# ...
cd ..
```

## ⚙️ Configuración

El archivo principal `ocr_inference.py` contiene un diccionario `CONFIG` con todos los parámetros configurables. Puedes modificarlo directamente o usar argumentos de línea de comandos.

### Parámetros principales

| Parámetro | Valor por defecto | Descripción |
|-----------|-------------------|-------------|
| `pdf_folder_path` | `"/home/cvasquez/olmOCR/PDFs"` | Carpeta con los PDFs a procesar |
| `reco_model` | `"DHiSS_v2_vitstr_base"` | Modelo de reconocimiento OCR |
| `detector_model` | `"db_resnet50"` | Modelo de detección de texto |
| `output_folder_name` | `"olmOCR_v2048"` | Nombre de la carpeta de salida |
| `use_custom_ocr` | `False` | Usar OCR personalizado para anchor text |
| `temperatures` | `[0.1, 0.4, 0.8]` | Temperaturas para fallback |
| `target_image_dim` | `2048` | Resolución de imagen (px) |
| `max_pixels` | `target_image_dim * 28 * 28` | Píxeles máximos (~1.6M) |
| `ocr_threshold` | `0.91` | Umbral de confianza OCR |
| `gpu_id` | `0` | ID de GPU a usar |
| `page_separator` | `False` | Agregar separadores entre páginas |

## 🚀 Uso

### Ejecución con Python

#### Uso básico (con valores por defecto)

```bash
python ocr_inference.py
```

**Valores por defecto:**
- Carpeta de PDFs: `/home/cvasquez/olmOCR/PDFs`
- Modelo: `DHiSS_v2_vitstr_base`
- Resolución: `2048px`
- Salida: `olmOCR_v2048/`
- OCR personalizado: Desactivado
- GPU: 0

#### Personalizar parámetros

```bash
# Cambiar carpeta de PDFs y nombre de salida
python ocr_inference.py \
    --pdf-folder /ruta/a/tus/pdfs \
    --output-folder mi_resultado

# Cambiar modelo y resolución
python ocr_inference.py \
    --reco-model DHiSS_v2_parseq \
    --target-dim 4096 \

# Activar OCR personalizado y cambiar GPU
python ocr_inference.py \
    --custom-ocr \
    --gpu 1

# Agregar separadores de página
python ocr_inference.py \
    --page-separator

# Configuración completa personalizada
python ocr_inference.py \
    --pdf-folder /datos/pdfs \
    --output-folder resultado_4k \
    --reco-model DHiSS_v2_vitstr_base \
    --target-dim 4096 \
    --custom-ocr \
    --temperatures 0.2 0.5 0.9 \
    --gpu 0 \
    --page-separator
```

#### Ver ayuda completa

```bash
python ocr_inference.py --help
```

### Ejecución con Scripts Bash

El proyecto incluye scripts bash para facilitar diferentes modos de ejecución:

#### 1. Ejemplo Simple (`run_simple_example.sh`)

Procesa una carpeta específica con una configuración predefinida:

```bash
# Editar el script para configurar la carpeta de PDFs
nano run_simple_example.sh

# Ejecutar
bash run_simple_example.sh
```

**Configuración en el script:**
```bash
PDF_FOLDER="/ruta/a/tus/pdfs"
python ocr_inference.py \
    --pdf-folder "$PDF_FOLDER" \
    --output-folder "test_darwin" \
    --target-dim 2048 \
    --reco-model "DHiSS_v2_vitstr_base" \
    --custom-ocr
```

#### 2. Múltiples Configuraciones (`run_multiple_configs.sh`)

Ejecuta el OCR con diferentes configuraciones automáticamente:

```bash
# Editar para configurar las pruebas deseadas
nano run_multiple_configs.sh

# Ejecutar
bash run_multiple_configs.sh
```

**Ejemplo de configuraciones:**
- Configuración 1: `DHiSS_v1_vitstr_base` @ 2048px
- Configuración 2: `DHiSS_v2_parseq` @ 2048px
- Configuración 3: `DHiSS_v2_vitstr_base` @ 4096px

#### 3. Procesamiento por Lotes (`run_batch_folders.sh`)

Procesa automáticamente todas las subcarpetas dentro de una carpeta principal:

```bash
# Editar para configurar la carpeta principal
nano run_batch_folders.sh

# Ejecutar
bash run_batch_folders.sh
```

**Estructura esperada:**
```
CARPETA_PRINCIPAL/
├── carpeta1/
│   ├── documento1.pdf
│   └── documento2.pdf
├── carpeta2/
│   ├── documento3.pdf
│   └── documento4.pdf
└── carpeta3/
    └── documento5.pdf
```

El script procesará cada subcarpeta automáticamente y generará un informe al final.

## 📁 Estructura del Proyecto

```
OCR_Nuestra_MemorIA/
├── ocr_inference.py           # Script principal de OCR
├── environment.yml            # Configuración del ambiente conda
├── requirements.txt           # Dependencias Python alternativas
├── README.md                  # Este archivo
├── .gitignore                 # Archivos a ignorar en git
│
├── ocr_weights/              # Pesos de los modelos (NO en repo)
│   ├── DHiSS_finetuning_parseq_10.pt
│   ├── DHiSS_finetuning_v2_parseq_10.pt
│   ├── DHiSS_finetuning_v2_vitstr_base_10.pt
│   └── DHiSS_finetuning_vitstr_base_10.pt
│
├── logs/                      # Logs de ejecución (generados)
│   ├── olmOCR_v2048.log
│   └── ...
│
├── run_simple_example.sh      # Script bash: ejemplo simple
├── run_multiple_configs.sh    # Script bash: múltiples configs
└── run_batch_folders.sh       # Script bash: procesamiento por lotes
```

## 📊 Logs y Resultados

### Ubicación de los Logs

Los logs se guardan automáticamente en la carpeta `logs/` en la raíz del proyecto:

```
logs/
├── olmOCR_v2048.log
├── test_darwin.log
└── mi_resultado.log
```

**Formato del nombre del log:**
- Por defecto: `<output_folder_name>.log`
- Personalizado: `--log-file mi_log.log`

### Contenido de los Logs

Los logs incluyen:
- Inicialización de modelos
- Progreso de procesamiento por página
- Texto de anclaje extraído
- Temperaturas utilizadas en fallback
- Errores y advertencias
- Resumen final de procesamiento

**Ejemplo:**
```
2025-11-12 14:30:15 - INFO - Configuration validation passed
2025-11-12 14:30:20 - INFO - Initializing Qwen2VL model and processor...
2025-11-12 14:30:45 - INFO - Model initialization completed successfully
2025-11-12 14:31:00 - INFO - Processing page 1...
2025-11-12 14:31:30 - INFO - Successfully extracted text from page 1 with temperature=0.1
```

### Ubicación de los Resultados

Los resultados se guardan en una subcarpeta dentro de la carpeta de PDFs procesados:

```
PDFs/
├── documento1.pdf
├── documento2.pdf
└── olmOCR_v2048/                    # Carpeta de salida
    ├── documento1_olmOCR_v2048.txt
    └── documento2_olmOCR_v2048.txt
```

**Formato del nombre:**
- `<nombre_pdf>_<output_folder_name>.txt`

### Monitoreo en Tiempo Real

Para ver los logs en tiempo real durante la ejecución:

```bash
# En otra terminal
tail -f logs/olmOCR_v2048.log
```

## 🤖 Modelos Soportados

El sistema soporta los siguientes modelos de reconocimiento OCR:

| Modelo | Checkpoint | Características |
|--------|------------|-----------------|
| `DHiSS_v1_parseq` | `DHiSS_finetuning_parseq_10.pt` | Modelo v1 con arquitectura PARSeq |
| `DHiSS_v1_sar_resnet31` | `DHiSS_finetuning_sar_resnet31_10.pt` | Modelo v1 con SAR ResNet31 |
| `DHiSS_v1_vitstr_base` | `DHiSS_finetuning_vitstr_base_10.pt` | Modelo v1 con ViTSTR base |
| `DHiSS_v2_vitstr_base` | `DHiSS_finetuning_v2_vitstr_base_10.pt` | **Modelo v2 con ViTSTR (recomendado)** |
| `DHiSS_v2_parseq` | `DHiSS_finetuning_v2_parseq_10.pt` | Modelo v2 con PARSeq |

### Recomendaciones

- **Para documentos generales**: `DHiSS_v2_vitstr_base` (por defecto)
- **Para alta precisión**: `DHiSS_v2_parseq` con resolución 4096px
- **Para velocidad**: `DHiSS_v1_vitstr_base` con resolución 2048px

### Detector

Todos los modelos usan `db_resnet50` como detector de texto por defecto.

## 🔍 Resolución de Problemas

### Error: "Model checkpoint not found"

```bash
# Verifica que los modelos estén en la carpeta correcta
ls -lh ocr_weights/
```

### Error: "CUDA out of memory"

Reduce la resolución o max_pixels:

```bash
python ocr_inference.py \
    --target-dim 1024 \
```

### Error: "PDF folder does not exist"

Verifica la ruta y ajústala:

```bash
python ocr_inference.py --pdf-folder /ruta/correcta/a/pdfs
```

## 📝 Licencia

[Especificar licencia aquí]

## 👥 Contribuciones

[Instrucciones para contribuir]

## 📧 Contacto

[Información de contacto]

---

**Nota**: Este proyecto está en desarrollo activo. Si encuentras problemas o tienes sugerencias, por favor abre un issue en el repositorio.
