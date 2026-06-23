/*
 * Copyright (c) 2020-2026 Kevin Day
 *
 * SPDX-License-Identifier: BSD-2-Clause-Patent
 * See LICENSE.md in the project root for full license text.
 */

package org.your.funnelcake;

/**
 * Thrown by a scaler constructor when the library reports a hard error
 * (a negative {@code FUSED_ERR_*} return code). Non-fatal conditions are
 * reported via {@link Warnings} instead, not as an exception.
 */
public final class FunnelcakeException extends RuntimeException {
    private final int code;

    FunnelcakeException(int code) {
        super(messageFor(code));
        this.code = code;
    }

    /** The negative {@code FUSED_ERR_*} code returned by the library. */
    public int code() {
        return code;
    }

    private static String messageFor(int code) {
        return switch (code) {
            case -1 -> "funnelcake: invalid flags (mixed families or unknown bits)";
            case -2 -> "funnelcake: no valid scale steps requested";
            case -3 -> "funnelcake: bad source dimensions";
            case -4 -> "funnelcake: source strides are not 32-byte aligned";
            default -> "funnelcake: error code " + code;
        };
    }
}
