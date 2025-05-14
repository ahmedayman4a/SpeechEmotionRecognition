# Speech Emotion Recognition - LaTeX Report

This directory contains the LaTeX source files for the Speech Emotion Recognition assignment report.

## Directory Structure

```
report/
├── images/            # All figures and plots
├── sections/          # LaTeX section files
├── build/             # Output directory for compiled PDF
├── main.tex           # Main LaTeX document
├── Makefile           # Compilation instructions
└── README.md          # This file
```

## Required Packages

### Installing LaTeX (Ubuntu/Debian)

If you don't have LaTeX installed, you can install the required packages with:

```bash
# Install basic TeX Live packages
sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra

# Install additional packages needed for this report
sudo apt-get install texlive-science texlive-pictures pgf
```

### Required LaTeX Packages

To compile this report, you need a LaTeX distribution with the following packages:

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
- algorithm, algpseudocode
- tikz

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

# Second pass: Resolve references
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

## Placeholder Images

The report is configured to compile even without actual image files. If you want to add your own images, place them in the `images/` directory with the following names:

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
- Add or modify figures in the `images/` directory 