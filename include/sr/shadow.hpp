#pragma once

#include <algorithm>

#include "sr/framebuffer.hpp"
#include "sr/math.hpp"

namespace sr {

// Orthographic, since a directional light's rays are parallel. Covers a sphere
// of the given radius around center.
[[nodiscard]] inline Mat4 directionalLightMatrix(const Vec3& lightDirection, const Vec3& center,
                                                 float radius, float zNear, float zFar) {
    const Vec3 direction = normalize(lightDirection);
    const Vec3 eye = center - direction * (radius * 2.0f);
    const Vec3 up = std::fabs(direction.y) > 0.95f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{0.0f, 1.0f, 0.0f};

    return orthographic(-radius, radius, -radius, radius, zNear, zFar) * lookAt(eye, center, up);
}

// Depth rendered from the light, plus the matrix that put it there. Fill it with
// a colorWrite = false pass, then sample it while shading.
class ShadowMap {
public:
    ShadowMap(int resolution, const Vec3& lightDirection, const Vec3& center, float radius)
        : depth_(resolution, resolution, DepthOnly::Yes),
          texelWorldSize_(2.0f * radius / static_cast<float>(resolution > 0 ? resolution : 1)),
          depthRange_(radius * 4.0f - kZNear) {
        lightViewProjection_ =
            directionalLightMatrix(lightDirection, center, radius, kZNear, radius * 4.0f);
        depth_.clearDepth(1.0f);
    }

    [[nodiscard]] Framebuffer& target() noexcept { return depth_; }
    [[nodiscard]] const Mat4& lightViewProjection() const noexcept { return lightViewProjection_; }

    // World units, except normalOffset which is in shadow texels. Keeping them in
    // world units is what stops a value tuned on one scene from causing acne on
    // the next when the light frustum is a different size.
    float depthBias = 0.02f;
    float slopeBias = 0.15f;
    float normalOffset = 3.0f;
    int pcfRadius = 1;  // 1 gives 3x3

    // 1 is lit, 0 is fully occluded.
    [[nodiscard]] float visibility(const Vec3& worldPosition, const Vec3& normal,
                                   float nDotL) const noexcept {
        // sin of the angle to the light. Acne gets worse with this, so the bias
        // terms scale by it.
        const float slope = std::sqrt(std::max(0.0f, 1.0f - nDotL * nDotL));

        // Moving the lookup off the surface stops it comparing against itself.
        const Vec3 samplePosition =
            worldPosition + normal * (texelWorldSize_ * normalOffset * (0.5f + slope));

        const Vec4 clip = lightViewProjection_ * Vec4(samplePosition, 1.0f);
        if (clip.w <= 0.0f) return 1.0f;

        const float invW = 1.0f / clip.w;
        const float u = clip.x * invW * 0.5f + 0.5f;
        const float v = 0.5f - clip.y * invW * 0.5f;
        const float z = clip.z * invW * 0.5f + 0.5f;

        // Outside the map counts as lit, so the map only has to cover the
        // casters and a large ground plane can extend well past it.
        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f || z > 1.0f) return 1.0f;

        const float bias = (depthBias + slopeBias * slope) / depthRange_;
        const int width = depth_.width();
        const int height = depth_.height();
        const int centerX = static_cast<int>(u * static_cast<float>(width));
        const int centerY = static_cast<int>(v * static_cast<float>(height));

        int lit = 0;
        int taps = 0;
        for (int dy = -pcfRadius; dy <= pcfRadius; ++dy) {
            for (int dx = -pcfRadius; dx <= pcfRadius; ++dx) {
                const int x = std::clamp(centerX + dx, 0, width - 1);
                const int y = std::clamp(centerY + dy, 0, height - 1);
                ++taps;
                if (z - bias <= depth_.depthAt(x, y)) ++lit;
            }
        }
        return static_cast<float>(lit) / static_cast<float>(taps);
    }

private:
    static constexpr float kZNear = 0.05f;

    Framebuffer depth_;
    Mat4 lightViewProjection_ = Mat4::identity();
    float texelWorldSize_ = 0.0f;
    float depthRange_ = 1.0f;
};

}  // namespace sr
