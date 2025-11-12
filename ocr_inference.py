#!/usr/bin/env python3
"""
OCR PDF Processor with OlmOCR and Custom DHiSS Models

This script processes PDF files using the OlmOCR model with custom DHiSS recognition models.
It includes a temperature-based fallback mechanism for robust text extraction.

Usage:
    python olmocr_run_better_fallback.py

Configuration:
    Edit the CONFIG dictionary below to customize:
    - PDF input folder path
    - OCR recognition model
    - Detector model
    - Output folder name
    - Custom OCR settings

Requirements:
    See requirements.txt for dependencies
    
Author: [Your Name]
Date: October 2025
"""

import torch
import base64
import json
import gc
import logging
import argparse
from pathlib import Path
from io import BytesIO
from typing import Optional, Tuple, List

import PyPDF2
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

from doctr.io import DocumentFile
from doctr.models import ocr_predictor, parseq, sar_resnet31, vitstr_base
from doctr.datasets import VOCABS

# ============================================================================
# CONFIGURATION PARAMETERS - CHANGE THESE VALUES AS NEEDED
# ============================================================================
CONFIG = {
    # PDF processing configuration
    "pdf_folder_path": "/home/cvasquez/olmOCR/PDFs",
    
    # OCR model configuration
    "reco_model": "DHiSS_v2_vitstr_base",  # Options: DHiSS_v1_parseq, DHiSS_v1_sar_resnet31, DHiSS_v1_vitstr_base, DHiSS_v2_vitstr_base, DHiSS_v2_parseq
    "detector_model": "db_resnet50",
    
    # Output configuration
    "output_folder_name": "olmOCR_v2048",  # Name of the folder where results will be saved
    "log_file_name": None,  # Log file name (None = uses output_folder_name + '.log')
    
    # Custom OCR configuration
    "use_custom_ocr": False,  # Set to False to use standard get_anchor_text
    
    # Temperature fallback strategy
    "temperatures": [0.1, 0.4, 0.8],  # List of temperatures to try in order
    "final_fallback_temperature": 0.1,  # Temperature for empty prompt fallback
    
    # Model parameters
    "max_new_tokens": 4096,
    "num_return_sequences": 2,
    
    # Image resolution parameters
    "target_image_dim": 2048,  # Longest side of the rendered image in pixels (1024, 1536, 2048, 4096)
    "min_pixels": None,   # Minimum pixels for image processing
    "max_pixels": None ,  # Maximum pixels for image processing - increase for higher resolution
    
    # OCR anchor parameters
    "ocr_threshold": 0.91,  # Confidence threshold for OCR word filtering
    
    # GPU configuration
    "gpu_id": 0,  # GPU device ID (0, 1, or 2)
    
    # Output formatting
    "page_separator": False,  # Add page separators (e.g., "Page (1)", "Page (2)") between pages
}

# Supported recognition models and their configurations
SUPPORTED_MODELS = {
    "DHiSS_v1_parseq": {
        "model_class": parseq,
        "checkpoint": "ocr_weights/DHiSS_finetuning_parseq_10.pt",
        "max_length": 50,
    },
    "DHiSS_v1_sar_resnet31": {
        "model_class": sar_resnet31,
        "checkpoint": "ocr_weights/DHiSS_finetuning_sar_resnet31_10.pt",
        "max_length": 50,
    },
    "DHiSS_v1_vitstr_base": {
        "model_class": vitstr_base,
        "checkpoint": "ocr_weights/DHiSS_finetuning_vitstr_base_10.pt",
        "max_length": 50,
    },
    "DHiSS_v2_vitstr_base": {
        "model_class": vitstr_base,
        "checkpoint": "ocr_weights/DHiSS_finetuning_v2_vitstr_base_10.pt",
        "max_length": 50,
    },
    "DHiSS_v2_parseq": {
        "model_class": parseq,
        "checkpoint": "ocr_weights/DHiSS_finetuning_v2_parseq_10.pt",
        "max_length": None,  # Uses default
    },
}
# ============================================================================


def parse_arguments():
    """
    Parse command line arguments to override CONFIG values.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='OCR PDF Processor with OlmOCR and DHiSS Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default CONFIG values:
  python olmocr_run_better_fallback.py
  
  # Override output folder and model:
  python olmocr_run_better_fallback.py --output-folder olmOCR_test --reco-model DHiSS_v2_parseq
  
  # Change resolution and PDF folder:
  python olmocr_run_better_fallback.py --pdf-folder /path/to/pdfs --target-dim 4096 --max-pixels 4063232
  
  # Disable custom OCR and change temperatures:
  python olmocr_run_better_fallback.py --no-custom-ocr --temperatures 0.2 0.5 0.9
        """
    )
    
    # PDF processing
    parser.add_argument('--pdf-folder', type=str, 
                        help=f"Path to folder containing PDFs (default: {CONFIG['pdf_folder_path']})")
    
    # OCR model configuration
    parser.add_argument('--reco-model', type=str, choices=list(SUPPORTED_MODELS.keys()),
                        help=f"Recognition model to use (default: {CONFIG['reco_model']})")
    parser.add_argument('--detector-model', type=str,
                        help=f"Detector model to use (default: {CONFIG['detector_model']})")
    
    # Output configuration
    parser.add_argument('--output-folder', type=str,
                        help=f"Name of output folder (default: {CONFIG['output_folder_name']})")
    parser.add_argument('--log-file', type=str,
                        help="Log file name (default: output_folder_name + '.log')")
    
    # Custom OCR
    parser.add_argument('--custom-ocr', dest='use_custom_ocr', action='store_true',
                        help="Enable custom OCR for anchor text")
    parser.add_argument('--no-custom-ocr', dest='use_custom_ocr', action='store_false',
                        help="Disable custom OCR for anchor text (use standard)")
    parser.set_defaults(use_custom_ocr=None)  # None means use CONFIG default
    
    # Temperature fallback
    parser.add_argument('--temperatures', type=float, nargs='+',
                        help=f"List of temperatures to try (default: {CONFIG['temperatures']})")
    parser.add_argument('--final-temperature', type=float,
                        help=f"Final fallback temperature (default: {CONFIG['final_fallback_temperature']})")
    
    # Model parameters
    parser.add_argument('--max-tokens', type=int,
                        help=f"Maximum new tokens to generate (default: {CONFIG['max_new_tokens']})")
    parser.add_argument('--num-sequences', type=int,
                        help=f"Number of sequences to return (default: {CONFIG['num_return_sequences']})")
    
    # Image resolution
    parser.add_argument('--target-dim', type=int,
                        help=f"Target image dimension in pixels (default: {CONFIG['target_image_dim']})")
    parser.add_argument('--min-pixels', type=int,
                        help=f"Minimum pixels for processing (default: {CONFIG['min_pixels']})")
    parser.add_argument('--max-pixels', type=int,
                        help=f"Maximum pixels for processing (default: {CONFIG['max_pixels']})")
    
    # OCR parameters
    parser.add_argument('--ocr-threshold', type=float,
                        help=f"OCR confidence threshold (default: {CONFIG['ocr_threshold']})")
    
    # GPU configuration
    parser.add_argument('--gpu', type=int, choices=[0, 1, 2],
                        help=f"GPU device ID to use (default: {CONFIG['gpu_id']})")
    
    # Output formatting
    parser.add_argument('--page-separator', dest='page_separator', action='store_true',
                        help="Add page separators (e.g., 'Page (1)', 'Page (2)') between pages")
    parser.add_argument('--no-page-separator', dest='page_separator', action='store_false',
                        help="Don't add page separators (default)")
    parser.set_defaults(page_separator=None)  # None means use CONFIG default
    
    return parser.parse_args()


def update_config_from_args(args):
    """
    Update CONFIG dictionary with values from command line arguments.
    
    Args:
        args: Parsed command line arguments
    """
    if args.pdf_folder is not None:
        CONFIG['pdf_folder_path'] = args.pdf_folder
    
    if args.reco_model is not None:
        CONFIG['reco_model'] = args.reco_model
    
    if args.detector_model is not None:
        CONFIG['detector_model'] = args.detector_model
    
    if args.output_folder is not None:
        CONFIG['output_folder_name'] = args.output_folder
    
    if args.log_file is not None:
        CONFIG['log_file_name'] = args.log_file
    
    if args.use_custom_ocr is not None:
        CONFIG['use_custom_ocr'] = args.use_custom_ocr
    
    if args.temperatures is not None:
        CONFIG['temperatures'] = args.temperatures
    
    if args.final_temperature is not None:
        CONFIG['final_fallback_temperature'] = args.final_temperature
    
    if args.max_tokens is not None:
        CONFIG['max_new_tokens'] = args.max_tokens
    
    if args.num_sequences is not None:
        CONFIG['num_return_sequences'] = args.num_sequences
    
    if args.target_dim is not None:
        CONFIG['target_image_dim'] = args.target_dim
    
    if args.min_pixels is not None:
        CONFIG['min_pixels'] = args.min_pixels
    
    if args.max_pixels is not None:
        CONFIG['max_pixels'] = args.max_pixels
    
    if args.ocr_threshold is not None:
        CONFIG['ocr_threshold'] = args.ocr_threshold
    
    if args.gpu is not None:
        CONFIG['gpu_id'] = args.gpu
    
    if args.page_separator is not None:
        CONFIG['page_separator'] = args.page_separator


def setup_logging(log_file: str = 'olmocr_out.log', enable_console: bool = False) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        log_file: Path to the log file
        enable_console: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    local_logger = logging.getLogger(__name__)
    local_logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    local_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    local_logger.addHandler(file_handler)
    
    # Console handler (optional)
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        local_logger.addHandler(console_handler)
    
    return local_logger


# Logger will be initialized in main() after parsing arguments
logger = None


def validate_config() -> bool:
    """
    Validate the configuration parameters.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    errors = []
    
    # Check PDF folder exists
    pdf_path = Path(CONFIG["pdf_folder_path"])
    if not pdf_path.exists():
        errors.append(f"PDF folder does not exist: {CONFIG['pdf_folder_path']}")
    
    # Check model is supported
    if CONFIG["reco_model"] not in SUPPORTED_MODELS:
        errors.append(f"Unsupported reco_model: {CONFIG['reco_model']}. "
                     f"Supported models: {list(SUPPORTED_MODELS.keys())}")
    
    # Check model checkpoint exists
    if CONFIG["reco_model"] in SUPPORTED_MODELS:
        checkpoint_path = Path(SUPPORTED_MODELS[CONFIG["reco_model"]]["checkpoint"])
        if not checkpoint_path.exists():
            errors.append(f"Model checkpoint not found: {checkpoint_path}")
    
    # Check temperatures are valid
    if not CONFIG["temperatures"] or not all(0 <= t <= 2 for t in CONFIG["temperatures"]):
        errors.append("Temperatures must be between 0 and 2")
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Configuration validation passed")
    return True
# ============================================================================


def initialize_model() -> Tuple[Qwen2VLForConditionalGeneration, AutoProcessor, torch.device]:
    """
    Initialize the Qwen2VL model and processor.
    
    Returns:
        Tuple of (model, processor, device)
    """
    logger.info("Initializing Qwen2VL model and processor...")
    
    gpu_id = CONFIG['gpu_id']
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", 
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu",
    ).eval()
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    logger.info("Model initialization completed successfully")
    return model, processor, device


def load_reco_model(reco: str):
    """
    Load a recognition model based on configuration.
    
    Args:
        reco: Model name from SUPPORTED_MODELS
        
    Returns:
        Loaded recognition model
    """
    if reco not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {reco}. Use one of {list(SUPPORTED_MODELS.keys())}")
    
    model_config = SUPPORTED_MODELS[reco]
    model_class = model_config["model_class"]
    checkpoint_path = model_config["checkpoint"]
    max_length = model_config["max_length"]
    
    # Initialize model with appropriate parameters
    if max_length:
        reco_model = model_class(
            pretrained=False, 
            pretrained_backbone=False, 
            vocab=VOCABS["spanish"], 
            max_length=max_length
        )
    else:
        reco_model = model_class(
            pretrained=False, 
            pretrained_backbone=False, 
            vocab=VOCABS["spanish"]
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    missing_keys, unexpected_keys = reco_model.load_state_dict(checkpoint, strict=False)
    
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint (ignored): {unexpected_keys}")
    if missing_keys:
        logger.warning(f"Missing keys in model (using random initialization): {missing_keys}")
    
    logger.info(f"Model {reco} loaded successfully from {checkpoint_path}")
    return reco_model


def initialize_ocr_model(reco: str = "DHiSS_v1_vitstr_base", detector: str = "db_resnet50"):
    """
    Initialize the OCR model with specified recognition and detection models.
    
    Args:
        reco: Recognition model name
        detector: Detection model name
        
    Returns:
        Initialized OCR predictor model
    """
    logger.info(f"Initializing OCR model with reco={reco}, detector={detector}...")
    
    # Load recognition model
    reco_model = load_reco_model(reco)
    
    # Initialize OCR predictor
    ocr_model = ocr_predictor(
        det_arch=detector,
        reco_arch=reco_model,
        det_bs=1,
        reco_bs=1,
        assume_straight_pages=False,
        straighten_pages=False,
        export_as_straight_boxes=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        detect_orientation=False,
        detect_language=False,
        disable_crop_orientation=False,
        disable_page_orientation=False,
        resolve_lines=True,
        resolve_blocks=False,
        paragraph_break=0.035,
        pretrained=True,
    ).eval()
    
    gpu_id = CONFIG['gpu_id']
    ocr_model.to(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"OCR model initialization completed with reco={reco} and detector={detector}")
    return ocr_model


def get_pdf_page_count(pdf_file_path: str) -> int:
    """
    Get the total number of pages in a PDF.
    
    Args:
        pdf_file_path: Path to the PDF file
        
    Returns:
        Number of pages in the PDF, or 0 if error
    """
    try:
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return len(pdf_reader.pages)
    except Exception as e:
        logger.error(f"Error reading PDF page count: {e}")
        return 0

def get_anchor_with_ocr_model(pdf_path: str, page_number: int, ocr_model, threshold: float = 0.95) -> str:
    """
    Extract anchor text using OCR model.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to process (1-indexed)
        ocr_model: Pre-initialized OCR model
        threshold: Confidence threshold for word filtering (default: 0.95)
        
    Returns:
        Formatted anchor text with word positions
    """
    with torch.no_grad():
        # Process PDF page
        doc = DocumentFile.from_pdf(pdf_path)
        result = ocr_model([doc[page_number - 1]])
        page = result.pages[0]
        
        # Initialize result with page dimensions
        page_result = f"Page dimensions: {page.dimensions[1]/2:.1f}x{page.dimensions[0]/2:.1f}\n"
        
        y, x = page.dimensions
        
        # Extract words with confidence above threshold
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.confidence > threshold:
                        # Calculate word geometry
                        ((xmin, ymin), (xmax, ymax)) = word.geometry
                        word_geometry = (
                            (int(xmin * x), int(ymin * y)),
                            (int(xmax * x), int(ymax * y))
                        )
                        (xmin_px, ymin_px), (xmax_px, ymax_px) = word_geometry
                        
                        # Calculate center of mass
                        mass_center = (
                            (xmin_px + xmax_px) // 2,
                            (ymin_px + ymax_px) // 2
                        )
                        mass_center = (
                            mass_center[0],
                            page.dimensions[0] - mass_center[1]
                        )
                        mass_center = (
                            int(mass_center[0] / 2),
                            int(mass_center[1] / 2)
                        )
                        
                        # Format: [XxY]word
                        result_string = f"[{mass_center[0]}x{mass_center[1]}]{word.value}\n"
                        page_result += result_string
        
        # Clean up
        del doc, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return page_result

def build_custom_ocr_prompt(anchor_text):
    """Build a custom prompt for OCR processing."""
    return (
        "Eres un experto modelo de OCR que extrae transcripciones de imágenes de documentos. "
        "Tu tarea es generar una transcripción precisa del texto visible en la imagen proporcionada. "
        "No incluyas información adicional, solo el texto visible en la imagen.\n\n"
        "Se te proporcionará un texto de anclaje extraído de la página del PDF. Esto lo puedes usar como referencia para la transcripción.\n\n"
        f"Texto de anclaje:\n{anchor_text}\n\n"
        "Por favor, extrae el texto y devuélvelo como un objeto JSON con la siguiente estructura:\n"
        "{\n"
        "  'natural_text': 'texto extraído aquí'"
        "}\n"
    )

def process_single_page(pdf_file_path: str, page_number: int, model, processor, device, 
                        use_custom_ocr: bool = True, ocr_model = None) -> str:
    """
    Process a single page of a PDF and return the extracted text.
    
    Args:
        pdf_file_path: Path to the PDF file
        page_number: Page number to process (1-indexed)
        model: Initialized Qwen2VL model
        processor: Model processor
        device: Torch device
        use_custom_ocr: Whether to use custom OCR for anchor text
        ocr_model: Pre-initialized OCR model (required if use_custom_ocr=True)
        
    Returns:
        Extracted text from the page
    """
    logger.info(f"Processing page {page_number}...")
    
    # Clear GPU cache at the start of each page
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get configuration parameters
    target_dim = CONFIG["target_image_dim"]
    ocr_threshold = CONFIG["ocr_threshold"]
    max_tokens = CONFIG["max_new_tokens"]
    num_sequences = CONFIG["num_return_sequences"]
    
    # Render page to image
    logger.info(f"Rendering PDF page {page_number} to base64 image...")
    image_base64 = render_pdf_to_base64png(
        pdf_file_path, page_number, target_longest_image_dim=target_dim
    )
    logger.info(f"PDF rendering completed for page {page_number}")

    # Extract anchor text
    logger.info(f"Extracting anchor text from page {page_number}...")
    if use_custom_ocr and ocr_model is not None:
        anchor_text = get_anchor_with_ocr_model(
            pdf_file_path, page_number, ocr_model, threshold=ocr_threshold
        )
        logger.info(f"Custom OCR anchor text extraction completed for page {page_number}")
    else:
        anchor_text = get_anchor_text(
            pdf_file_path, page_number, pdf_engine="pdfreport", target_length=4000
        )
        logger.info(f"Standard anchor text extraction completed for page {page_number}")

    # Build prompt
    prompt = build_finetuning_prompt(anchor_text)
    #logger.info(f"Prompt built for page {page_number}: {prompt[:100]}")  # Log first 100 characters of prompt
    logger.info(f"Prompt built for page {page_number}: {prompt}")  # Log first 100 characters of prompt
    
    def _attempt_generation(prompt_text, temperature=0.1):
        """Helper function to attempt text generation with given prompt and temperature."""
        with torch.no_grad():  # Disable gradient computation
            # Get resolution parameters from CONFIG
            min_pixels = CONFIG["min_pixels"] or 256 * 28 * 28
            max_pixels = CONFIG["max_pixels"] or CONFIG["target_image_dim"] * 28 * 28
            
            # Construct messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels,
                            }
                        },
                    ],
                }
            ]

            # Process inputs
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

            inputs = processor(
                text=[text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for (key, value) in inputs.items()}

            prompt_length = inputs["input_ids"].shape[1]
            logger.info(f"Prompt length for page {page_number}: {prompt_length}")

            # Generate output
            logger.info(f"Generating OCR output for page {page_number} with temperature={temperature}...")
            output = model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_tokens,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_return_sequences=num_sequences,
                do_sample=True,
            )

            # Decode output
            new_tokens = output[:, prompt_length:]
            logger.info(f"New tokens generated for page {page_number}: {new_tokens.shape}")
            text_output = processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            
            # Explicitly delete tensors to free memory
            del inputs, output, new_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return text_output

    # Try with different temperatures from CONFIG
    temperatures = CONFIG["temperatures"]
    
    for temp in temperatures:
        try:
            logger.info(f"Attempting generation for page {page_number} with temperature={temp}")
            text_output = _attempt_generation(prompt, temperature=temp)
            json_output = json.loads(text_output[0])
            logger.info(f"JSON output for page {page_number}: {json_output}")
            # Ensure natural_text is always a string (avoid None propagating)
            natural_text = json_output.get("natural_text")
            if not isinstance(natural_text, str):
                natural_text = "" if natural_text is None else str(natural_text)
            logger.info(f"Successfully extracted text from page {page_number} with temperature={temp}")
            
            # Clean up variables
            del text_output, json_output
            gc.collect()  # Force garbage collection
            
            return natural_text
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON output for page {page_number} with temperature={temp}: {e}")
            if temp == temperatures[-1]:
                # Last temperature attempt failed, proceed to empty prompt fallback
                logger.info(f"All temperature attempts failed for page {page_number}. Retrying with empty prompt...")
            else:
                logger.info(f"Retrying page {page_number} with higher temperature...")
            # Clean up and continue to next temperature
            if 'text_output' in locals():
                del text_output
            gc.collect()
            continue
        except Exception as e:
            logger.error(f"Error during generation for page {page_number} with temperature={temp}: {e}")
            if temp == temperatures[-1]:
                # Last temperature attempt failed, proceed to empty prompt fallback
                logger.info(f"All temperature attempts failed for page {page_number}. Retrying with empty prompt...")
            else:
                logger.info(f"Retrying page {page_number} with higher temperature...")
            # Clean up and continue to next temperature
            gc.collect()
            continue
    
    # Final fallback: empty prompt with configured fallback temperature
    try:
        fallback_temp = CONFIG["final_fallback_temperature"]
        logger.info(f"Attempting final fallback with empty prompt and temperature={fallback_temp} for page {page_number}")
        text_output = _attempt_generation("", temperature=fallback_temp)
        if type(text_output) is list and len(text_output) > 0:
            text_output = text_output[0]
        logger.info(f"Output for page {page_number} (empty prompt fallback): {text_output}")
        logger.info(f"Successfully extracted text from page {page_number} on empty prompt fallback")
        
        # Clean up and return
        result = text_output
        del text_output
        gc.collect()
        return result
    except Exception as e:
        logger.error(f"Error during empty prompt fallback for page {page_number}: {e}")
        # If all attempts fail, return empty string
        return ""
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def process_pdf(pdf_file_path, model, processor, device, use_custom_ocr=True, ocr_model=None):
    """Process all pages of a PDF and return combined text."""
    logger.info(f"Starting to process PDF: {pdf_file_path}")
    
    # Get total page count
    total_pages = get_pdf_page_count(pdf_file_path)
    if total_pages == 0:
        logger.error(f"Could not determine page count for {pdf_file_path}")
        return ""
    
    logger.info(f"PDF has {total_pages} pages")
    
    all_pages_text = []
    use_page_separator = CONFIG.get('page_separator', False)
    
    for page_num in range(1, total_pages + 1):
        try:
            page_text = process_single_page(pdf_file_path, page_num, model, processor, device, use_custom_ocr, ocr_model)
            # Guard against None or non-string page outputs
            if not isinstance(page_text, str):
                page_text = "" if page_text is None else str(page_text)
            
            # Add page separator if enabled
            if use_page_separator:
                page_text = f"Page ({page_num})\n{page_text}"
            
            all_pages_text.append(page_text)
            logger.info(f"Page {page_num}/{total_pages} completed")
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            # Add empty string or separator for failed pages
            if use_page_separator:
                all_pages_text.append(f"Page ({page_num})\n[Error processing page]")
            else:
                all_pages_text.append("")
    
    # Combine all pages with newline separator
    combined_text = "\n".join(t if isinstance(t, str) else ("" if t is None else str(t)) for t in all_pages_text)
    #combined_text = "".join(all_pages_text)
    logger.info(f"Successfully processed all {total_pages} pages")
    
    return combined_text


def save_output(pdf_file_path, extracted_text, output_folder_name=None):
    """Save the extracted text to a file."""
    if output_folder_name is None:
        output_folder_name = CONFIG["output_folder_name"]
    
    pdf_name = Path(pdf_file_path).stem
    
    # Create subfolder for outputs
    output_dir = Path(pdf_file_path).parent / output_folder_name
    output_dir.mkdir(exist_ok=True)

    output_filename = f"{pdf_name}_{output_folder_name}.txt"
    output_path = output_dir / output_filename
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        logger.info(f"Output saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving output file: {e}")
        return None


def main():
    """Main function to orchestrate the OCR process."""
    global logger
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Update CONFIG with command line arguments
    update_config_from_args(args)
    
    # Initialize logger after CONFIG is updated
    log_filename = CONFIG.get("log_file_name") or f"{CONFIG['output_folder_name']}.log"
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Save log file in logs directory
    log_file_path = logs_dir / log_filename
    logger = setup_logging(log_file=str(log_file_path))
    
    # Print header
    print("=" * 80)
    print("OCR PDF Processor with OlmOCR and DHiSS Models".center(80))
    print("=" * 80)
    print()
    
    # Validate configuration
    logger.info("Validating configuration...")
    if not validate_config():
        log_file = CONFIG.get("log_file_name") or f"{CONFIG['output_folder_name']}.log"
        print(f"❌ Configuration validation failed. Check logs/{log_file} for details.")
        return
    
    print("✅ Configuration validated successfully")
    print(f"📁 PDF folder: {CONFIG['pdf_folder_path']}")
    print(f"🤖 Recognition model: {CONFIG['reco_model']}")
    print(f"🔍 Detector: {CONFIG['detector_model']}")
    print(f"💾 Output folder: {CONFIG['output_folder_name']}")
    print(f"🖼️  Image resolution: {CONFIG['target_image_dim']}px (min: {CONFIG['min_pixels']:,}, max: {CONFIG['max_pixels']:,})")
    print()
    
    # Get configuration from CONFIG dictionary
    use_custom_ocr = CONFIG["use_custom_ocr"]
    reco_model = CONFIG["reco_model"]
    detector_model = CONFIG["detector_model"]
    pdf_folder_path = CONFIG["pdf_folder_path"]
    output_folder_name = CONFIG["output_folder_name"]
    
    # Initialize models
    print("🔄 Initializing models...")
    model, processor, device = initialize_model()
    
    # Initialize OCR model if using custom OCR
    ocr_model = None
    if use_custom_ocr:
        ocr_model = initialize_ocr_model(reco=reco_model, detector=detector_model)
    
    print("✅ Models initialized successfully")
    print()
    
    # Get all PDF files in the folder
    pdf_files = list(Path(pdf_folder_path).glob("*.pdf"))
    logger.info(f"PDFs found: {pdf_files}")
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_folder_path}")
        print(f"❌ No PDF files found in {pdf_folder_path}")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF file(s) to process")
    print("-" * 80)
    
    # Statistics tracking
    successful = 0
    failed = 0
    total_pages = 0
    
    # Process each PDF file
    for i, pdf_file_path in enumerate(pdf_files, 1):
        logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_file_path.name}")
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file_path.name}")
        
        try:
            # Get page count
            page_count = get_pdf_page_count(str(pdf_file_path))
            total_pages += page_count
            print(f"    Pages: {page_count}")
            
            # Process the entire PDF
            extracted_text = process_pdf(str(pdf_file_path), model, processor, device, use_custom_ocr, ocr_model)
            
            if extracted_text:
                # Save output
                output_path = save_output(str(pdf_file_path), extracted_text, output_folder_name)
                if output_path:
                    logger.info(f"OCR completed for {pdf_file_path.name}. Output saved to: {output_path}")
                    print(f"    ✅ Completed - saved to: {output_path.name}")
                    successful += 1
                else:
                    logger.error(f"Failed to save output file for {pdf_file_path.name}")
                    print(f"    ❌ Failed to save output file")
                    failed += 1
            else:
                logger.error(f"No text extracted from {pdf_file_path.name}")
                print(f"    ❌ No text extracted")
                failed += 1
                
        except Exception as e:
            logger.error(f"Error processing {pdf_file_path.name}: {e}")
            print(f"    ❌ Error: {e}")
            failed += 1
            continue
    
    # Print summary
    print()
    print("=" * 80)
    print("PROCESSING SUMMARY".center(80))
    print("=" * 80)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total pages: {total_pages}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {(successful/len(pdf_files)*100):.1f}%")
    print("=" * 80)
    
    logger.info("All PDFs processing completed")
    logger.info(f"Summary: {successful} successful, {failed} failed out of {len(pdf_files)} PDFs")
    
    # Get log filename from CONFIG
    log_file = CONFIG.get("log_file_name") or f"{CONFIG['output_folder_name']}.log"
    print(f"\n📝 Detailed logs saved to: logs/{log_file}")


if __name__ == "__main__":
    main()