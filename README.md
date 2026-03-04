# SAM-Guided-OVD: Open-Vocabulary DETR with Class-Agnostic Proposals for Grounded Robotic VQA

This repository contains the official implementation of the Master's Thesis: **Enhancing Open-Vocabulary DETR with Class-Agnostic SAM Proposals for Grounded Robotic VQA.**

## Abstract
Standard Open-Vocabulary Object Detectors (OV-OD) often struggle with fine-grained domain shifts, such as identifying highly specific, novel industrial tools in robotic assembly environments. This project introduces a unified architecture that leverages the geometric precision of Fast Segment Anything (FastSAM) as a dynamic Region Proposal Network (RPN) for a Denoising DETR backbone (OV-DQUO). By injecting deterministic bounding box priors into the DETR decoder, the model guarantees the localization of novel objects, subsequently generating highly accurate semantic features used for Grounded Visual Question Answering (VQA).

## Architecture
1. **Geometric Prior Generation:** FastSAM generates dense, class-agnostic bounding box proposals.
2. **Dynamic Query Formulation:** Proposals are injected as `reference_points` into the OV-DQUO decoder.
3. **Open-Vocabulary Classification:** Denoising text queries strictly classify the isolated objects.
4. **Grounded VQA:** Stabilized object queries cross-attend with text tokens to answer operator prompts.

## Setup & Installation
*(To be updated as development progresses)*

## Dataset Preparation
*(To be updated: Instructions for the GAI20 Industrial Dataset)*