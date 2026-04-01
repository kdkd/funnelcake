#include <stdarg.h>
#include <stdio.h>

#include "log.h"
#include "funnelcake.h"

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
