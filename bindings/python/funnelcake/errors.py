# Copyright (c) 2020-2026 Kevin Day
#
# SPDX-License-Identifier: BSD-2-Clause-Patent
# See LICENSE.md in the project root for full license text.

"""Error and warning types."""

_MESSAGES = {
    -1: "invalid flags (mixed families or unknown bits)",
    -2: "no valid scale steps requested",
    -3: "bad source dimensions",
    -4: "source strides are not 32-byte aligned",
}


class FunnelcakeError(Exception):
    """Raised by a scaler constructor on a hard error (negative FUSED_ERR_*)."""

    def __init__(self, code: int):
        self.code = code
        msg = _MESSAGES.get(code, f"error code {code}")
        super().__init__(f"funnelcake: {msg}")


class Warnings:
    """Composable non-fatal conditions reported by a successful init."""

    __slots__ = ("bits",)

    def __init__(self, bits: int):
        self.bits = bits

    def scalar(self) -> bool:
        """A step fell back to the scalar kernel."""
        return bool(self.bits & (1 << 0))

    def partial(self) -> bool:
        """At least one requested step was rejected."""
        return bool(self.bits & (1 << 1))

    def cropped(self) -> bool:
        """The source was cropped to fit."""
        return bool(self.bits & (1 << 2))

    def perfect(self) -> bool:
        """Every requested output was produced with SIMD and no cropping."""
        return self.bits == 0

    def __repr__(self) -> str:
        return (
            f"Warnings(scalar={self.scalar()}, partial={self.partial()}, "
            f"cropped={self.cropped()})"
        )
