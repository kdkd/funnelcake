/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

#include <stdarg.h>
#include <stdio.h>

#include "log.h"
#include "internal.h"
#include "funnelcake.h"

/* Scratch pool exhaustion is an init-time sizing invariant violation: the
 * pool is sized from the same parameters the kernel uses, so this must never
 * trigger in correct code. If it does, the affected frame's output is invalid;
 * emit one loud diagnostic rather than corrupting silently. */
void fused_scratch_exhausted_warn(void)
{
    static int warned = 0;
    if (!warned) {
        warned = 1;
        fprintf(stderr,
            "funnelcake: internal error: scratch pool exhausted - "
            "output for this frame is invalid\n");
    }
}

void fused_log(const fused_log_config_t *config, int level, const char *fmt, ...)
{
    va_list ap;

    /* NULL config: fall back to stderr */
    if (config == NULL) {
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        return;
    }

    switch (config->target) {

    case FUSED_LOG_SUPPRESS:
        /* Discard silently */
        return;

    case FUSED_LOG_STDOUT:
        va_start(ap, fmt);
        vfprintf(stdout, fmt, ap);
        va_end(ap);
        break;

    case FUSED_LOG_FILE:
        if (config->file != NULL) {
            va_start(ap, fmt);
            vfprintf(config->file, fmt, ap);
            va_end(ap);
        }
        break;

    case FUSED_LOG_CALLBACK:
        if (config->callback != NULL) {
            char buf[1024];
            va_start(ap, fmt);
            vsnprintf(buf, sizeof(buf), fmt, ap);
            va_end(ap);
            config->callback(level, buf, config->callback_ctx);
        }
        break;

    case FUSED_LOG_STDERR:
    default:
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        break;
    }
}
