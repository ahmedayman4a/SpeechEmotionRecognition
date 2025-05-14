# Makefile for Speech Emotion Recognition LaTeX Report

# Source files
MAIN=main
BIB=references
TEX_FILES=$(MAIN).tex $(wildcard sections/*.tex)

# Output directory
OUTPUT_DIR=build

# Commands
LATEXMK=latexmk
PDFLATEX=pdflatex

# Options
LATEXMK_OPTS=-pdf -silent -outdir=$(OUTPUT_DIR)

# Default target
all: $(OUTPUT_DIR)/$(MAIN).pdf

# Create output directory if it doesn't exist
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Generate PDF using latexmk
$(OUTPUT_DIR)/$(MAIN).pdf: $(TEX_FILES) | $(OUTPUT_DIR)
	$(LATEXMK) $(LATEXMK_OPTS) $(MAIN)

# Alternative manual compilation procedure
manual: $(TEX_FILES)| $(OUTPUT_DIR)
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN)
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN)
	$(PDFLATEX) -output-directory=$(OUTPUT_DIR) $(MAIN)

# Clean all generated files
clean:
	rm -rf $(OUTPUT_DIR)

# Clean auxiliary files but keep PDF
clean-aux:
	$(LATEXMK) -c -outdir=$(OUTPUT_DIR) $(MAIN)

.PHONY: all clean clean-aux manual 