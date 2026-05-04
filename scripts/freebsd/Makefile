PORTNAME=	funnelcake
DISTVERSION=	0.1.0
CATEGORIES=	multimedia

MAINTAINER=	toasty@dragondata.com
COMMENT=	SIMD YUV scaler with HDR/SDR tonemapping
WWW=		https://github.com/kdkd/funnelcake

# BSD-2-Clause-Patent is not in the framework's _LICENSE_LIST, so it is
# declared as a custom license here. The text is reproduced from
# LICENSE.md (SPDX: BSD-2-Clause-Patent).
LICENSE=	BSD2CLAUSEPATENT
LICENSE_NAME=	BSD 2-Clause Plus Patent License
LICENSE_FILE=	${WRKSRC}/LICENSE.md
LICENSE_PERMS=	dist-mirror dist-sell pkg-mirror pkg-sell auto-accept

ONLY_FOR_ARCHS=		aarch64 amd64
ONLY_FOR_ARCHS_REASON=	requires AVX2 (amd64) or NEON (aarch64) SIMD kernels

USES=		gmake
USE_GITHUB=	yes
GH_ACCOUNT=	kdkd
# Tag a release upstream and bump DISTVERSION above; GH_TAGNAME defaults
# to ${DISTVERSION}, so no GH_TAGNAME line is needed for tagged releases.

USE_LDCONFIG=	yes

OPTIONS_DEFINE=		FFMPEG
OPTIONS_DEFAULT=
FFMPEG_DESC=		Install ffmpeg for benchmark comparisons against swscale
FFMPEG_RUN_DEPENDS=	ffmpeg:multimedia/ffmpeg

# The upstream Makefile honors PREFIX, LIBDIR, INCLUDEDIR, PKGCONFDIR,
# DESTDIR, and INSTALL. FreeBSD ports put pkg-config files under
# ${PREFIX}/libdata/pkgconfig, not ${PREFIX}/lib/pkgconfig.
MAKE_ARGS=	PREFIX=${PREFIX} \
		LIBDIR=${PREFIX}/lib \
		INCLUDEDIR=${PREFIX}/include \
		PKGCONFDIR=${PREFIX}/libdata/pkgconfig \
		VERSION=${DISTVERSION} \
		INSTALL="${INSTALL} ${COPY}"

ALL_TARGET=	all shared

PLIST_FILES=	include/funnelcake.h \
		lib/libfunnelcake.a \
		lib/libfunnelcake.so \
		lib/libfunnelcake.so.1 \
		libdata/pkgconfig/funnelcake.pc

.include <bsd.port.mk>
