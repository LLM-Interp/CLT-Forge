# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

* Added circuit-tracer CLT loading support for HuggingFace, local safetensors, and circuit-tracer cache sources in attribution workflows.
* Added conversion utilities for saving circuit-tracer attribution graphs and feature metadata in the existing CLT-Forge visual interface format.
* Added a notebook showing how to load open-source circuit-tracer CLTs and visualize them with the CLT-Forge interface.

## [0.1.0] - 2026-02-16

### Added

* Initial open-source release of the CLT library
* Training pipeline with configurable runners
* Feature sharding, ddp, sfdp implemented
* Compression implemented in activation store (not yet tested properly on downstream effects)

### Notes

* This is the first public release and the API may change in future versions.
