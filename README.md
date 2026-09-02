# Antimicrobial Resistance Prediction Using Nucleotide Transformer

## Overview

This project aims to predict antimicrobial resistance (AMR) from bacterial genome sequences.

The system takes a bacterial genome sequence in `.fna` format, processes the DNA sequence using a pretrained Nucleotide Transformer, and generates a fixed-size 512-dimensional genome embedding.

These embeddings are then intended to be used as input to a neural-network classifier that predicts resistance across multiple antibiotics.

## Project Pipeline

Bacterial Genome (.fna)
        ->
Sequence Preprocessing
        ->
Nucleotide Transformer v2
        ->
Genome Windows
        ->
512-dimensional Embeddings
        ->
Neural Network Classifier
        ->
AMR Predictions
        ->
12 Antibiotics
