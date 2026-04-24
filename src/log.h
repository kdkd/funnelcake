#ifndef FUNNELCAKE_LOG_H
#define FUNNELCAKE_LOG_H

#include "funnelcake.h"

/*
 * fused_log - emit a formatted message through the given logging config.
 *
 * config  : logging configuration from the scaler context. NULL means stderr.
 * level   : FUSED_LOG_ERROR or FUSED_LOG_WARN (passed through to callbacks).
 * fmt     : printf-style format string followed by arguments.
 *
 * Dispatch rules:
 *   FUSED_LOG_STDERR   (or NULL config) - fprintf to stderr
 *   FUSED_LOG_STDOUT                    - fprintf to stdout
 *   FUSED_LOG_FILE                      - fprintf to config->file (if non-NULL)
 *   FUSED_LOG_SUPPRESS                  - discard, return immediately
 *   FUSED_LOG_CALLBACK                  - format into a 1024-byte buffer,
 *                                         call config->callback(level, buf, ctx)
 */
void fused_log(const fused_log_config_t *config, int level, const char *fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;

#endif /* FUNNELCAKE_LOG_H */
