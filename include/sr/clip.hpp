// Sutherland-Hodgman clipping in homogeneous clip space, before the perspective
// divide. Vertices behind the eye have w <= 0, and dividing by that folds them
// back into view, so they have to go before the divide happens.
#pragma once

#include <array>

#include "sr/math.hpp"

namespace sr {

template <class V>
struct ClipVertex {
    Vec4 position;
    V varyings{};
};

// Seven planes can each add at most one vertex to a triangle.
inline constexpr int kMaxClippedVertices = 12;

template <class V>
using ClipPolygon = std::array<ClipVertex<V>, kMaxClippedVertices>;

enum class ClipPlane { PositiveW, Left, Right, Bottom, Top, Near, Far };

inline constexpr std::array<ClipPlane, 7> kClipPlanes = {
    ClipPlane::PositiveW, ClipPlane::Left, ClipPlane::Right, ClipPlane::Bottom,
    ClipPlane::Top,       ClipPlane::Near, ClipPlane::Far};

// Vertices are kept where this is >= 0.
[[nodiscard]] inline float planeDistance(ClipPlane plane, const Vec4& p) noexcept {
    constexpr float kMinW = 1e-5f;  // keeps the perspective divide finite
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

// Whole triangle outside one plane. Worth checking first, it is the common case.
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

// Clips in place and returns the surviving vertex count, or 0 if fully rejected.
// scratch is caller-owned so the hot path does not allocate.
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

            // Clip space is still linear here, so varyings blend with the same t.
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
