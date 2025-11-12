#!/bin/bash
# ============================================================================
# Script de Procesamiento por Lotes: Recorre subcarpetas con PDFs
# ============================================================================
# Este script procesa automáticamente todas las subcarpetas dentro de una
# carpeta principal, ejecutando el OCR para cada grupo de PDFs.
#
# Estructura esperada:
#   CARPETA_PRINCIPAL/
#   ├── carpeta1/
#   │   ├── documento1.pdf
#   │   └── documento2.pdf
#   ├── carpeta2/
#   │   ├── documento3.pdf
#   │   └── documento4.pdf
#   └── carpeta3/
#       └── documento5.pdf
# ============================================================================

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# CONFIGURACIÓN - MODIFICA ESTOS VALORES
# ============================================================================

# Carpeta principal que contiene las subcarpetas con PDFs
MAIN_FOLDER="/home/cvasquez/olmOCR/MMDH_Repositorio_Archivo Extranjería"

# Script de Python a ejecutar
PYTHON_SCRIPT="olmocr_run_better_fallback.py"

# Configuración del modelo OCR
RECO_MODEL="DHiSS_v2_vitstr_base"
TARGET_DIM=2048

# Opciones adicionales
USE_CUSTOM_OCR="--custom-ocr"
PAGE_SEPARATOR="--page-separator"  # Cambiar a "--no-page-separator" si no quieres separadores
GPU_ID="--gpu 0"  # Cambiar a --gpu 1 o --gpu 2 si tienes múltiples GPUs

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

echo "============================================================================"
echo -e "${GREEN}Script de Procesamiento por Lotes - OlmOCR${NC}"
echo "============================================================================"
echo ""
echo -e "${BLUE}Configuración:${NC}"
echo "  Carpeta principal: $MAIN_FOLDER"
echo "  Modelo: $RECO_MODEL"
echo "  Resolución: ${TARGET_DIM}px"
echo "============================================================================"
echo ""

# Verificar que la carpeta principal existe
if [ ! -d "$MAIN_FOLDER" ]; then
    echo -e "${RED}❌ Error: La carpeta principal no existe: $MAIN_FOLDER${NC}"
    exit 1
fi

# Verificar que el script de Python existe
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}❌ Error: No se encuentra el script: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Contador de carpetas procesadas
total_folders=0
successful_folders=0
failed_folders=0

# Array para guardar carpetas que fallaron
declare -a failed_folder_list

# Obtener el timestamp de inicio
start_time=$(date +%s)

echo -e "${YELLOW}Escaneando subcarpetas...${NC}"
echo ""

# Recorrer todas las subcarpetas en la carpeta principal
for subfolder in "$MAIN_FOLDER"/*/; do
    # Verificar que sea un directorio
    if [ ! -d "$subfolder" ]; then
        continue
    fi
    
    # Obtener el nombre de la carpeta (sin la ruta completa)
    folder_name=$(basename "$subfolder")
    
    # Contar PDFs en la subcarpeta
    pdf_count=$(find "$subfolder" -maxdepth 1 -name "*.pdf" -type f | wc -l)
    
    # Si no hay PDFs, saltar esta carpeta
    if [ "$pdf_count" -eq 0 ]; then
        echo -e "${YELLOW}⚠️  Saltando '$folder_name' (no contiene PDFs)${NC}"
        continue
    fi
    
    total_folders=$((total_folders + 1))
    
    echo "----------------------------------------------------------------------------"
    echo -e "${BLUE}📁 Procesando carpeta [$total_folders]: $folder_name${NC}"
    echo -e "${BLUE}   PDFs encontrados: $pdf_count${NC}"
    echo "----------------------------------------------------------------------------"
    
    # Crear nombre de carpeta de salida
    output_folder_name="olmOCR_${folder_name}_${RECO_MODEL}_${TARGET_DIM}px"
    
    # Ejecutar el script de Python
    echo -e "${GREEN}▶️  Iniciando procesamiento...${NC}"
    
    if python "$PYTHON_SCRIPT" \
        --pdf-folder "$subfolder" \
        --output-folder "$output_folder_name" \
        --target-dim "$TARGET_DIM" \
        --reco-model "$RECO_MODEL" \
        $USE_CUSTOM_OCR \
        $PAGE_SEPARATOR \
        $GPU_ID; then
        
        echo -e "${GREEN}✅ Carpeta '$folder_name' procesada exitosamente${NC}"
        successful_folders=$((successful_folders + 1))
    else
        echo -e "${RED}❌ Error procesando carpeta '$folder_name'${NC}"
        failed_folders=$((failed_folders + 1))
        failed_folder_list+=("$folder_name")
    fi
    
    echo ""
done

# Calcular tiempo total
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))

# Resumen final
echo "============================================================================"
echo -e "${GREEN}RESUMEN DEL PROCESAMIENTO${NC}"
echo "============================================================================"
echo "Total de carpetas procesadas: $total_folders"
echo -e "${GREEN}✅ Exitosas: $successful_folders${NC}"
echo -e "${RED}❌ Fallidas: $failed_folders${NC}"
echo ""
echo "Tiempo total: ${hours}h ${minutes}m ${seconds}s"

# Mostrar lista de carpetas fallidas si hay alguna
if [ "$failed_folders" -gt 0 ]; then
    echo ""
    echo -e "${RED}Carpetas que fallaron:${NC}"
    for folder in "${failed_folder_list[@]}"; do
        echo -e "${RED}  - $folder${NC}"
    done
fi

echo "============================================================================"

# Exit code basado en el éxito
if [ "$failed_folders" -eq 0 ]; then
    echo -e "${GREEN}¡Todos los procesamientos completados exitosamente! 🎉${NC}"
    exit 0
else
    echo -e "${YELLOW}Procesamiento completado con algunos errores ⚠️${NC}"
    exit 1
fi
