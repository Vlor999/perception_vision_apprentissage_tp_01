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

