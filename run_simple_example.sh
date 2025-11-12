#!/bin/bash
# ============================================================================
# Ejemplo Simple: Ejecutar con diferentes resoluciones
# ============================================================================

# Configurar la carpeta de PDFs
PDF_FOLDER="/home/cvasquez/olmOCR/MMDH_Repositorio_Archivo Extranjería/COL.1"
PYTHON_SCRIPT="ocr_inference.py"
echo -e "${BLUE}Ejecutando con resolución (4096px)...${NC}"
python $PYTHON_SCRIPT \
    --pdf-folder "$PDF_FOLDER" \
    --output-folder "test_darwin" \
    --target-dim 2048 \
    --reco-model "DHiSS_v2_vitstr_base" \
    --custom-ocr \

echo "¡Todos los procesamiento completados!"
