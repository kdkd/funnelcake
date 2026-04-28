# Changelog

All notable changes to funnelcake are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project does
not yet follow a tagged release cadence, so changes accumulate under
**Unreleased** until a tag is cut.

## [Unreleased]

### Added
- `FUSED_ERR_OUT_OF_MEMORY` (-5) hard-error code returned by
  `fused_scaler_init` and `fused_hdr_init` when allocation of the internal
  state struct fails. Previously these paths returned the misleading
  `FUSED_ERR_NO_STEPS`.
- `FUSED_LOG_INFO` (2) log level for low-frequency status / diagnostic
  messages. Routed through the existing `log_warnings` config so callers
  that install a `FUSED_LOG_CALLBACK` can filter info out by inspecting
  the `level` argument while still receiving warnings.

### Changed
- The "tone map LUTs generated" diagnostic in `fused_tonemap_generate_luts`
  now logs at `FUSED_LOG_INFO` instead of `FUSED_LOG_WARN`. Callers that
  previously suppressed it via `log_warnings = FUSED_LOG_SUPPRESS` keep
  the same behavior; callback-based loggers can now keep warnings while
  dropping info.
- The "no SIMD support detected" notice from both init functions now
  routes through `fused_log(&ctx->log_warnings, FUSED_LOG_WARN, …)`
  instead of writing to `stderr` directly, so users with a configured
  log target see it where they expect.
- `fused_scaler_free` and `fused_hdr_free` now also reset
  `effective_width` and `effective_height` on the context, matching the
  reset of the other result fields.

### Fixed
- **HDR init memory leak**: `fused_hdr_init` could leak `state->sdr_temp[i]`
  buffers if SDR-only steps had been allocated successfully and a later
  step (the 1:1 tonemap output, or the "no valid steps" check) failed.
  The error paths called `fused_hdr_free(ctx)` while `ctx->_internal` was
  still NULL, skipping the cleanup of `state`'s sdr_temp pointers, and
  then `free(state)` released the struct without freeing those buffers.
  Init now attaches `state` to `ctx->_internal` immediately after
  allocation so any subsequent error path goes through `fused_hdr_free`
  and releases everything.
- The misaligned-source warning emitted by `fused_scaler_run` and
  `fused_hdr_run` used a process-wide `static int warned` flag, so the
  first context to encounter a misaligned source silenced the warning
  for every other context in the process (and the flag was not
  thread-safe). Each context now owns its own `src_misaligned_warned`
  flag inside its internal state.
- The misaligned-source warning was missing a trailing newline.

### Documentation
- `docs/API.md`: documented `FUSED_LOG_INFO`, `FUSED_ERR_OUT_OF_MEMORY`,
  and the relationship between log levels and the routing config.

---

## Conventions

- **Added** — new public API surface (functions, constants, struct fields).
- **Changed** — non-breaking behavioral changes to existing API.
- **Deprecated** — APIs scheduled for removal.
- **Removed** — APIs that have been deleted.
- **Fixed** — bug fixes that don't change documented behavior.
- **Security** — vulnerability fixes.
- **Performance** — measurable speed/memory wins, with a one-line summary
  of the workload and the delta.
- **Documentation** — doc-only changes worth noting.

Group breaking changes under their own **Breaking** subsection and call
out the impact on callers.
