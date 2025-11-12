# OCR PDF Processor with OlmOCR

Este script procesa archivos PDF usando el modelo OlmOCR con modelos de reconocimiento DHiSS personalizados. Incluye un mecanismo de fallback basado en temperatura para extracción robusta de texto.

## 📋 Características

- ✅ Procesamiento por lotes de PDFs
- ✅ Modelos DHiSS personalizados para español
- ✅ Fallback inteligente con múltiples temperaturas
- ✅ Logging detallado para debugging
- ✅ Configuración centralizada y fácil de modificar
- ✅ Manejo robusto de errores

## 🚀 Instalación

### 1. Requisitos del sistema
- Python 3.8+
- CUDA compatible GPU (recomendado)
- 16GB+ RAM
- 10GB+ espacio en disco

### 2. Instalar dependencias

```bash
# Crear entorno virtual (recomendado)
conda create -n olmocr python=3.10
conda activate olmocr

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Descargar modelos

Asegúrate de tener los pesos de los modelos en la carpeta `ocr_weights/`:
```
ocr_weights/
├── DHiSS_finetuning_parseq_10.pt
├── DHiSS_finetuning_sar_resnet31_10.pt
├── DHiSS_finetuning_vitstr_base_10.pt
├── DHiSS_finetuning_v2_vitstr_base_10.pt
└── DHiSS_finetuning_v2_parseq_10.pt
```

## ⚙️ Configuración

### Configuración principal

Edita el diccionario `CONFIG` en `olmocr_run_better_fallback.py`:

```python
CONFIG = {
    # Ruta a la carpeta con PDFs
    "pdf_folder_path": "/ruta/a/tus/PDFs",
    
    # Modelo de reconocimiento
    # Opciones: DHiSS_v1_parseq, DHiSS_v1_sar_resnet31, 
    #           DHiSS_v1_vitstr_base, DHiSS_v2_vitstr_base, DHiSS_v2_parseq
    "reco_model": "DHiSS_v2_parseq",
    
    # Modelo detector
    "detector_model": "db_resnet50",
    
    # Nombre de carpeta para resultados
    "output_folder_name": "olmOCR_DHiSS_v2_parseq_OCR",
    
    # Usar OCR personalizado (True recomendado)
    "use_custom_ocr": True,
    
    # Temperaturas para fallback (de más conservador a más creativo)
    "temperatures": [0.1, 0.4, 0.8],
    
    # Temperatura para fallback final con prompt vacío
    "final_fallback_temperature": 0.1,
    
    # Parámetros del modelo
    "max_new_tokens": 4096,
    "num_return_sequences": 2,
    "target_image_dim": 1024,
    
    # Umbral de confianza para OCR
    "ocr_threshold": 0.91,
}
```

### Modelos disponibles

| Modelo | Descripción | Recomendado para |
|--------|-------------|------------------|
| `DHiSS_v1_parseq` | ParseQ v1 | Textos generales |
| `DHiSS_v1_sar_resnet31` | SAR ResNet v1 | Documentos escaneados |
| `DHiSS_v1_vitstr_base` | ViTSTR v1 | Textos impresos |
| `DHiSS_v2_vitstr_base` | ViTSTR v2 (mejorado) | Textos impresos mejorado |
| `DHiSS_v2_parseq` | ParseQ v2 (mejorado) | **Recomendado general** |

## 📖 Uso

### Procesar todos los PDFs en una carpeta

1. Configura la ruta en `CONFIG["pdf_folder_path"]`
2. Ejecuta:

```bash
python olmocr_run_better_fallback.py
```

Los resultados se guardarán en:
```
/ruta/PDFs/
└── olmOCR_DHiSS_v2_parseq_OCR/
    ├── documento1_olmOCR_DHiSS_v2_parseq_OCR.txt
    ├── documento2_olmOCR_DHiSS_v2_parseq_OCR.txt
    └── ...
```

El script procesará automáticamente todos los PDFs encontrados en la carpeta configurada.
## 📊 Estrategia de Fallback

El script implementa un sistema de fallback en 4 niveles:

1. **Intento 1**: Temperatura 0.1 (más conservador/determinista)
2. **Intento 2**: Temperatura 0.4 (moderado)
3. **Intento 3**: Temperatura 0.8 (más creativo)
4. **Fallback final**: Prompt vacío con temperatura 0.1

Si todos fallan, retorna cadena vacía.

## 📝 Logs

Los logs se guardan automáticamente en `olmocr_out.log`:

```
2025-10-22 10:30:15 - INFO - Configuration validation passed
2025-10-22 10:30:20 - INFO - Initializing Qwen2VL model and processor...
2025-10-22 10:31:00 - INFO - Using device: cuda
2025-10-22 10:31:05 - INFO - Processing page 1...
2025-10-22 10:31:30 - INFO - Successfully extracted text from page 1 with temperature=0.1
```

## 🐛 Troubleshooting

### Error: "PDF folder does not exist"
Verifica que la ruta en `CONFIG["pdf_folder_path"]` existe y es correcta.

### Error: "Model checkpoint not found"
Asegúrate de tener los archivos `.pt` en la carpeta `ocr_weights/`.

### Error: CUDA out of memory
Reduce `max_new_tokens` o `target_image_dim` en CONFIG.

### Calidad OCR baja
- Prueba diferentes modelos (`reco_model`)
- Ajusta `ocr_threshold` (valores más bajos = más palabras, menos precisión)
- Prueba con `use_custom_ocr: False` para comparar

## 🔧 Personalización avanzada

### Cambiar temperaturas de fallback

```python
CONFIG = {
    ...
    "temperatures": [0.05, 0.2, 0.5, 0.9],  # 4 intentos
    "final_fallback_temperature": 0.05,
}
```

### Habilitar logging en consola

Modifica la llamada a `setup_logging()`:

```python
logger = setup_logging(enable_console=True)
```

### Usar OCR estándar en vez de DHiSS

```python
CONFIG = {
    ...
    "use_custom_ocr": False,
}
```

## 📦 Estructura de archivos

```
olmOCR/
├── olmocr_run_better_fallback.py  # Script principal
├── olmocr_out.log                 # Logs de ejecución
├── README_olmocr_run.md          # Esta documentación
├── requirements.txt               # Dependencias
├── ocr_weights/                   # Pesos de modelos DHiSS
│   └── *.pt
└── PDFs/                          # PDFs a procesar
    ├── documento1.pdf
    └── olmOCR_DHiSS_v2_parseq_OCR/  # Resultados
        └── *.txt
```

## 👥 Para colaboradores

Si vas a pasar este código a otra persona:

1. **Compartir**:
   - `olmocr_run_better_fallback.py`
   - `README_olmocr_run.md`
   - `requirements.txt`
   - Carpeta `ocr_weights/` (si tienen los permisos)

2. **Instrucciones básicas**:
   - Instalar dependencias: `pip install -r requirements.txt`
   - Editar solo el diccionario `CONFIG`
   - Ejecutar: `python olmocr_run_better_fallback.py`

3. **Validación**:
   - El script valida automáticamente la configuración
   - Muestra errores claros si falta algo

## 📄 Licencia

[Especificar licencia si aplica]

## ✉️ Contacto

[Tu información de contacto]

---
**Última actualización**: Octubre 2025
