#!/bin/bash
# ============================================================================
# Script para ejecutar olmocr_run_better_fallback.py con múltiples configuraciones
# ============================================================================

# Colores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "  Ejecutando OCR con múltiples configuraciones"
echo "========================================================================"
echo ""

# Ruta al script Python
PYTHON_SCRIPT="ocr_inference.py"

# Carpeta base de PDFs
PDF_FOLDER="/home/cvasquez/olmOCR/PDFs"

# ============================================================================
# Configuración 1: DHiSS_v1_vitstr_base con resolución (2048px)
# ============================================================================
echo -e "${BLUE}[1/3] Ejecutando con resolución (2048px)...${NC}"
python $PYTHON_SCRIPT \
    --pdf-folder "$PDF_FOLDER" \
    --output-folder "olmOCR_DHiSS_v1_vitstr_base_2048px" \
    --target-dim 2048 \
    --reco-model "DHiSS_v1_vitstr_base" \
    --custom-ocr \

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Configuración 1 completada${NC}\n"
else
    echo -e "${RED}✗ Configuración 1 falló${NC}\n"
fi

# ============================================================================
# Configuración 2: DhiSS_v2_parseq con alta resolución (2048px)
# ============================================================================
echo -e "${BLUE}[2/3] Ejecutando con alta resolución (2048px)...${NC}"
python $PYTHON_SCRIPT \
    --pdf-folder "$PDF_FOLDER" \
    --output-folder "olmOCR_DHiSS_v2_parseq_2048px" \
    --target-dim 2048 \
    --reco-model "DHiSS_v2_parseq" \
    --custom-ocr \

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Configuración 2 completada${NC}\n"
else
    echo -e "${RED}✗ Configuración 2 falló${NC}\n"
fi

# ============================================================================
# Configuración 3: DHiSS_v2_vitstr_base con custom OCR y alta resolución (4096px)
# ============================================================================
echo -e "${BLUE}[3/3] Ejecutando DHiSS_v2_vitstr_base con alta resolución (4096px)...${NC}"
python $PYTHON_SCRIPT \
    --pdf-folder "$PDF_FOLDER" \
    --output-folder "olmOCR_DHiSS_v2_vitstr_base_4096px" \
    --target-dim 4096 \
    --max-pixels 3211264 \
    --reco-model "DHiSS_v2_vitstr_base" \
    --custom-ocr \

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Configuración 3 completada${NC}\n"
else
    echo -e "${RED}✗ Configuración 3 falló${NC}\n"
fi

# ============================================================================
# Resumen
# ============================================================================
echo "========================================================================"
echo "  Todas las configuraciones han sido ejecutadas"
echo "========================================================================"
echo ""
echo "Revisa los logs en:"
echo "  - logs/olmOCR_DHiSS_v1_vitstr_base_2048px.log"
echo "  - logs/olmOCR_DHiSS_v2_parseq_2048px.log"
echo "  - logs/olmOCR_DHiSS_v2_vitstr_base_4096px.log"
echo ""
echo "Resultados en:"
echo "  - $PDF_FOLDER/olmOCR_DHiSS_v1_vitstr_base_2048px/"
echo "  - $PDF_FOLDER/olmOCR_DHiSS_v2_parseq_2048px/"
echo "  - $PDF_FOLDER/olmOCR_DHiSS_v2_vitstr_base_4096px/"
