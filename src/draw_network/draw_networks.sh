#!/bin/bash

# Array of files to process (without .py extension)
INPUT_DIR="archs"
files=("draw_deeper_detector" "draw_resnet_detector" "draw_simple_detector")

# Get the project root directory (two levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR" && pwd )"
OUTPUT_DIR="$PROJECT_ROOT/network_diagrams"

# Create network_diagrams directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each file
for filename in "${files[@]}"; do
    echo "Processing $filename..."
    
    # Generate LaTeX file
    file_path="${INPUT_DIR}/${filename}"
    echo "Running: python3 ${file_path}.py"
    
    # Change to the directory containing the Python file to ensure relative paths work
    cd "${INPUT_DIR}"
    python3 "${filename}.py"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to generate LaTeX for $filename"
        cd ..
        continue
    fi
    
    # Check if LaTeX file was generated (in current directory - archs/)
    if [ ! -f "${filename}.tex" ]; then
        echo "Error: LaTeX file ${filename}.tex was not generated"
        cd ..
        continue
    fi
    
    # Move LaTeX file to parent directory for compilation
    mv "${filename}.tex" "../${filename}.tex"
    cd ..
    
    # Compile to PDF
    echo "Compiling: pdflatex ${filename}.tex"
    pdflatex "${filename}.tex"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to compile PDF for $filename"
        continue
    fi
    
    # Move PDF to network_diagrams
    if [ -f "${filename}.pdf" ]; then
        mv "${filename}.pdf" "$OUTPUT_DIR/"
        echo "Moved ${filename}.pdf to $OUTPUT_DIR"
    else
        echo "No file named: ${filename}.pdf"
    fi
    
    # Clean up temporary files
    rm -f "${filename}.aux" "${filename}.log" "${filename}.vscodeLog" "${filename}.tex"
    
done

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

