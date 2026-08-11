// Sutherland-Hodgman polygon clipping in homogeneous clip space.
//
// A triangle is clipped against the seven half-spaces that define the visible
// volume, before the perspective divide. Doing it here (rather than in NDC) is
// what keeps geometry crossing the eye plane from wrapping around: those
// vertices have w <= 0, and dividing by w would fold them back into view.
#pragma once

#include <array>

#include "sr/math.hpp"

namespace sr {

// A vertex mid-pipeline: clip-space position plus whatever the vertex shader
// wants interpolated. `V` only has to support `V * float` and `V + V`.
template <class V>
struct ClipVertex {
    Vec4 position;
    V varyings{};
};

// Clipping a triangle by 7 planes can add at most one vertex per plane.
inline constexpr int kMaxClippedVertices = 12;

template <class V>
using ClipPolygon = std::array<ClipVertex<V>, kMaxClippedVertices>;

// Signed distance to each half-space; a vertex is kept where this is >= 0.
enum class ClipPlane { PositiveW, Left, Right, Bottom, Top, Near, Far };

inline constexpr std::array<ClipPlane, 7> kClipPlanes = {
    ClipPlane::PositiveW, ClipPlane::Left, ClipPlane::Right, ClipPlane::Bottom,
    ClipPlane::Top,       ClipPlane::Near, ClipPlane::Far};

[[nodiscard]] inline float planeDistance(ClipPlane plane, const Vec4& p) noexcept {
    // Guard w against exactly zero so the perspective divide stays finite.
    constexpr float kMinW = 1e-5f;
    switch (plane) {
        case ClipPlane::PositiveW: return p.w - kMinW;
        case ClipPlane::Left:      return p.x + p.w;
        case ClipPlane::Right:     return p.w - p.x;
        case ClipPlane::Bottom:    return p.y + p.w;
        case ClipPlane::Top:       return p.w - p.y;
        case ClipPlane::Near:      return p.z + p.w;
        case ClipPlane::Far:       return p.w - p.z;
    }
    return 0.0f;
}

// True when the whole triangle sits outside one plane, which is the common case
// worth rejecting before doing any clipping work.
template <class V>
[[nodiscard]] bool triviallyRejected(const ClipVertex<V>& a, const ClipVertex<V>& b,
                                     const ClipVertex<V>& c) noexcept {
    for (ClipPlane plane : kClipPlanes) {
        if (planeDistance(plane, a.position) < 0.0f && planeDistance(plane, b.position) < 0.0f &&
            planeDistance(plane, c.position) < 0.0f) {
            return true;
        }
    }
    return false;
}

// True when every vertex is inside every plane, so clipping can be skipped.
template <class V>
[[nodiscard]] bool triviallyAccepted(const ClipVertex<V>& a, const ClipVertex<V>& b,
                                     const ClipVertex<V>& c) noexcept {
    for (ClipPlane plane : kClipPlanes) {
        if (planeDistance(plane, a.position) < 0.0f || planeDistance(plane, b.position) < 0.0f ||
            planeDistance(plane, c.position) < 0.0f) {
            return false;
        }
    }
    return true;
}

// Clips `poly` (with `count` vertices) in place against every frustum plane and
// returns the surviving vertex count, or 0 if the polygon was fully rejected.
// `scratch` is caller-owned so the hot path performs no allocation.
template <class V>
int clipPolygon(ClipPolygon<V>& poly, int count, ClipPolygon<V>& scratch) noexcept {
    for (ClipPlane plane : kClipPlanes) {
        if (count < 3) return 0;

        int outCount = 0;
        for (int i = 0; i < count; ++i) {
            const ClipVertex<V>& current = poly[static_cast<std::size_t>(i)];
            const ClipVertex<V>& next = poly[static_cast<std::size_t>((i + 1) % count)];

            const float dCurrent = planeDistance(plane, current.position);
            const float dNext = planeDistance(plane, next.position);
            const bool currentInside = dCurrent >= 0.0f;
            const bool nextInside = dNext >= 0.0f;

            if (currentInside && outCount < kMaxClippedVertices) {
                scratch[static_cast<std::size_t>(outCount++)] = current;
            }

            // Crossing the plane: emit the intersection. Interpolating in clip
            // space is linear, so varyings can be blended with the same t.
            if (currentInside != nextInside && outCount < kMaxClippedVertices) {
                const float t = dCurrent / (dCurrent - dNext);
                ClipVertex<V> hit;
                hit.position = current.position + (next.position - current.position) * t;
                hit.varyings = current.varyings * (1.0f - t) + next.varyings * t;
                scratch[static_cast<std::size_t>(outCount++)] = hit;
            }
        }

        for (int i = 0; i < outCount; ++i) {
            poly[static_cast<std::size_t>(i)] = scratch[static_cast<std::size_t>(i)];
        }
        count = outCount;
    }
    return count < 3 ? 0 : count;
}

}  // namespace sr
