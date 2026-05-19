# Repository Guidelines

## Project Structure & Module Organization
必须用中文和我交流。这个仓库是用来对这个插件进行二次开发，以满足我们自己的需要。
dev_docs目录是用来放计划文档跟随进度的。你会和claude code协同工作，请合理使用说明文档交流。
This repository contains a 3D Slicer extension plus a Python server for nnInteractive segmentation.

- `slicer_plugin/SlicerNNInteractive/` contains the scripted Slicer module; `SlicerNNInteractive.py` is the main client logic.
- `Resources/` stores Qt UI files and icons, and `Testing/` stores Slicer regression tests plus reference NIfTI masks in `Testing/Data/`.
- `server/` contains the FastAPI server package, Dockerfile, and package metadata.
- `img/` stores documentation images.

## Build, Test, and Development Commands

- `cmake -S . -B build` configures the extension project. Use a Slicer-compatible CMake environment when building as an extension.
- `cd server && docker build -t nninteractive_slicer_server .` builds the server container.
- `cd server && docker run -p 1527:1527 --gpus all --rm -d nninteractive_slicer_server` starts the server container on port `1527`.
- `cd server && uv run nninteractive-slicer-server --host 0.0.0.0 --port 1527` runs the packaged server locally when dependencies are available.

For extension development, open 3D Slicer, use Extension Wizard, and select `slicer_plugin`.

## Coding Style & Naming Conventions

Python is the primary language. Follow PEP 8 with 4-space indentation, descriptive names, and module constants in `UPPER_SNAKE_CASE`. Keep Slicer widget, logic, and test classes named after the module, for example `SlicerNNInteractiveSegmentationTest`. Preserve Qt object names and resource paths when editing `.ui` files or icons.

## Testing Guidelines

Regression tests are Slicer Python tests in `slicer_plugin/SlicerNNInteractive/Testing/Python/`. Test files should use the `*Test.py` suffix and be registered in `Testing/CMakeLists.txt`.

To run the suite, start the nnInteractive server, launch Slicer, enable Developer Mode, open `Self Tests`, select `SlicerNNInteractive`, and run `Reload and Test`. Reference masks are checked against `Testing/Data/`. Only regenerate references with `SLICER_NNI_GENERATE_TEST_MASK=1` after manual review.

## Commit & Pull Request Guidelines

Git history was unavailable due to a safe-directory ownership check, so use clear, imperative commit subjects such as `Fix server URL validation` or `Add lasso regression case`. Keep commits focused.

Pull requests should include a concise description, linked issues, Slicer and server versions used for testing, test results, and screenshots or recordings for UI changes. Follow `CONTRIBUTING.md` and never report security-sensitive issues publicly.

## Security & Configuration Tips

Do not commit model weights, credentials, local environment files, or generated medical data. The server expects GPU-capable PyTorch; document CUDA, Python, and Slicer version assumptions when changing setup instructions.
