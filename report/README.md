# Speech Emotion Recognition - LaTeX Report

This directory contains the LaTeX source files for the Speech Emotion Recognition assignment report.

## Directory Structure

```
report/
├── images/            # All figures and plots
├── sections/          # LaTeX section files
├── tables/            # Table data/code (if separate from main sections)
├── main.tex           # Main LaTeX document
├── references.bib     # Bibliography file
├── Makefile           # Compilation instructions
└── README.md          # This file
```

## Required Packages

To compile this report, you need to have a LaTeX distribution installed (TeX Live, MikTeX, or MacTeX), which includes the following packages:

- graphicx
- amsmath, amssymb, amsfonts
- booktabs
- multirow
- float
- hyperref
- subcaption
- xcolor
- geometry
- listings
- csquotes
- biblatex (with bibtex backend)

## Compilation Instructions

### Using the Makefile (Recommended)

1. Make sure you have `latexmk` installed (included in most LaTeX distributions)
2. Run:
   ```
   make
   ```
3. The compiled PDF will be available in the `build/` directory

### Manual Compilation Alternative

If you don't have `latexmk` or prefer manual compilation:

```bash
# Create build directory
mkdir -p build

# First pass: Process LaTeX
pdflatex -output-directory=build main

# Process bibliography
bibtex build/main

# Second pass: Incorporate bibliography
pdflatex -output-directory=build main

# Third pass: Resolve references
pdflatex -output-directory=build main
```

You can also use the Makefile for this approach:
```bash
make manual
```

### Cleaning Up

To remove all generated files:
```bash
make clean
```

To clean auxiliary files but keep the PDF:
```bash
make clean-aux
```

## Required Images

Before compiling, ensure that the following images are placed in the `images/` directory:

- waveforms.png
- 1d_features.png
- mel_spectrograms.png
- 1d_cnn_architecture.png
- 2d_cnn_architecture.png
- combined_model_architecture.png
- activation_lr_heatmap.png
- confusion_matrix.png
- training_curves.png

## Customizing the Report

- Edit author names and title in `main.tex`
- Modify individual sections in the `sections/` directory
- Add references in `references.bib`
- Add or modify figures in the `images/` directory 