# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/`, with the public API in `include/funnelcake.h`. Architecture-specific kernels are split by backend (`src/kernels_avx2.c`, `src/kernels_neon.c`, `src/kernels_scalar.c`). Tests live in `test/` and build into a single `funnelcake_test` binary. Reference docs are in `docs/`, and build and tuning notes are in `README.md` and `INSTALL.md`.

## Build, Test, and Development Commands
Use `make` from the repository root.

- `make`: builds `libfunnelcake.a`.
- `make test`: builds and runs validation and correctness tests.
- `make bench`: runs benchmark mode with `./funnelcake_test --bench`.
- `make visual`: writes visual test PNGs to `output/`.
- `make clean`: removes objects, test binaries, and generated output.
- `make LTO=1 TUNE=native`: enables link-time optimization and CPU-specific tuning.
- `make pgo LTO=1 TUNE=native`: runs the profile-guided optimization flow.

## Coding Style & Naming Conventions
This is a C11 codebase built with `-Wall -Wextra -Werror`; new code should compile warning-free under both the library and test targets. Follow the existing style: 4-space indentation, K&R braces, `snake_case` for functions and variables, and `UPPER_SNAKE_CASE` for macros and flags such as `FUSED_SCALE_4X`. Keep public declarations in `include/` and internal helpers in `src/`. There is no formatter configured, so match nearby files closely.

## Testing Guidelines
Add or extend tests in `test/test_*.c` when changing validation rules, output dimensions, kernel behavior, or public API semantics. Keep helper declarations in `test/test_main.h` or `test/test_patterns.h` when shared across test files. Run `make test` before submitting; use `make bench` for performance-sensitive changes and `make visual` when touching image output logic.

## Commit & Pull Request Guidelines
Current history uses short, imperative commit subjects such as `Add README.md`. Continue with concise subjects under about 72 characters and keep each commit focused. Pull requests should describe the behavioral change, list the commands you ran (`make test`, `make bench`, etc.), and note any platform-specific validation such as AVX2 or NEON coverage. Include benchmark deltas or sample output notes when performance or visual behavior changes.
