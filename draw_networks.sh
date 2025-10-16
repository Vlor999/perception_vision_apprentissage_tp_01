#!/bin/bash

# Array of files to process (without .py extension)
INPUT_DIR="src/draw_network/archs"
files=("draw_deeper_detector" "draw_resnet_detector" "draw_simple_detector")

# Get the project root directory (script is at project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
OUTPUT_DIR="$PROJECT_ROOT/src/draw_network/network_diagrams"

# Create network_diagrams directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each file
for filename in "${files[@]}"; do
    echo "Processing $filename..."
    
    # Full path to the Python file
    file_path="${PROJECT_ROOT}/${INPUT_DIR}/${filename}.py"
    echo "Running: python3 ${file_path}"
    
    # Run the Python script from the project root to maintain relative paths
    cd "$PROJECT_ROOT"
    python3 "${file_path}"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate LaTeX for $filename"
        continue
    fi
    
    # Check if LaTeX file was generated in the archs directory
    tex_file="${PROJECT_ROOT}/${INPUT_DIR}/${filename}.tex"
    if [ ! -f "$tex_file" ]; then
        echo "Error: LaTeX file ${filename}.tex was not generated"
        continue
    fi
    
    # Move to src/draw_network for compilation
    cd "${PROJECT_ROOT}/src/draw_network"
    mv "archs/${filename}.tex" "${filename}.tex"
    
    # Compile to PDF
    echo "Compiling: pdflatex ${filename}.tex"
    pdflatex "${filename}.tex"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to compile PDF for $filename"
        continue
    fi
    
    # Move PDF to network_diagrams
    if [ -f "${filename}.pdf" ]; then
        mv "${filename}.pdf" "network_diagrams/"
        echo "Moved ${filename}.pdf to network_diagrams/"
        
        # Convert PDF to PNG with high quality
        echo "Converting ${filename}.pdf to PNG..."
        cd "network_diagrams"
        
        # Check if ImageMagick magick is available (newer versions)
        if command -v magick >/dev/null 2>&1; then
            magick -density 300 "${filename}.pdf" -quality 90 "${filename}.png"
            echo "Generated ${filename}.png using ImageMagick"
        # Check if ImageMagick convert is available (older versions)
        elif command -v convert >/dev/null 2>&1; then
            convert -density 300 "${filename}.pdf" -quality 90 "${filename}.png"
            echo "Generated ${filename}.png using ImageMagick (legacy convert command)"
        # Check if pdftoppm is available (from poppler-utils)
        elif command -v pdftoppm >/dev/null 2>&1; then
            pdftoppm -png -r 300 "${filename}.pdf" "${filename}"
            # pdftoppm generates filename-1.png, rename to filename.png
            if [ -f "${filename}-1.png" ]; then
                mv "${filename}-1.png" "${filename}.png"
                echo "Generated ${filename}.png using pdftoppm"
            fi
        # Check if sips is available (macOS built-in)
        elif command -v sips >/dev/null 2>&1; then
            sips -s format png "${filename}.pdf" --out "${filename}.png"
            echo "Generated ${filename}.png using sips"
        else
            echo "Warning: No PDF to PNG converter found. Install ImageMagick, poppler-utils, or use macOS sips"
        fi
        
        cd ..
    else
        echo "No file named: ${filename}.pdf"
    fi
    
    # Clean up temporary files
    rm -f "${filename}.aux" "${filename}.log" "${filename}.vscodeLog" "${filename}.tex"
    
done

# Return to project root
cd "$PROJECT_ROOT"

echo ""
echo "All network diagrams generated successfully!"
echo "PDFs saved to: $OUTPUT_DIR"
echo "PNGs saved to: $OUTPUT_DIR"

# Open all PDFs (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ls "$OUTPUT_DIR"/*.pdf 1> /dev/null 2>&1; then
        open "$OUTPUT_DIR"/*.pdf
    else
        echo "No PDF files found to open"
    fi
else
    # Linux
    for filename in "${files[@]}"; do
        if [ -f "$OUTPUT_DIR/${filename}.pdf" ]; then
            xdg-open "$OUTPUT_DIR/${filename}.pdf" 2>/dev/null &
        fi
    done
fi

