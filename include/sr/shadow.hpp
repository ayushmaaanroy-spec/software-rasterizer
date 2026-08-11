// Shadow mapping, which falls out of the pipeline almost for free: render the
// scene from the light with colour writes off, and the depth buffer you already
// have becomes a record of the nearest occluder in every direction.
#pragma once

#include <algorithm>

#include "sr/framebuffer.hpp"
#include "sr/math.hpp"

namespace sr {

// A light view-projection covering a sphere of `radius` around `center`.
// Orthographic, because a directional light's rays are parallel.
[[nodiscard]] inline Mat4 directionalLightMatrix(const Vec3& lightDirection, const Vec3& center,
                                                 float radius, float zNear, float zFar) {
    const Vec3 direction = normalize(lightDirection);
    const Vec3 eye = center - direction * (radius * 2.0f);
    // Any up vector works as long as it is not parallel to the light.
    const Vec3 up = std::fabs(direction.y) > 0.95f ? Vec3{0.0f, 0.0f, 1.0f} : Vec3{0.0f, 1.0f, 0.0f};

    return orthographic(-radius, radius, -radius, radius, zNear, zFar) * lookAt(eye, center, up);
}

class ShadowMap {
public:
    // The map is fitted to a bounding sphere of the casters. Knowing `radius`
    // is what lets the biases below be specified in world units: expressed in
    // normalised depth they would silently change meaning every time the light
    // frustum was resized, and a value tuned on one scene would produce acne on
    // the next.
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

    // All three are in world units except the last, which is in shadow texels.
    float depthBias = 0.02f;         // fights depth quantisation
    float slopeBias = 0.15f;         // scales up as the surface turns edge-on
    float normalOffset = 3.0f;       // shifts the lookup off the surface
    int pcfRadius = 1;               // 1 gives 3x3 percentage-closer filtering

    // Fraction of the light reaching the point: 1 lit, 0 fully occluded.
    [[nodiscard]] float visibility(const Vec3& worldPosition, const Vec3& normal,
                                   float nDotL) const noexcept {
        // sin of the angle between the surface and the light. Everything that
        // causes acne gets worse with this, so every bias term scales by it.
        const float slope = std::sqrt(std::max(0.0f, 1.0f - nDotL * nDotL));

        // Moving the lookup off the surface along its normal is the cheapest
        // acne fix there is: it shifts the comparison to a point the occluder
        // test cannot confuse with the surface itself.
        const Vec3 samplePosition =
            worldPosition + normal * (texelWorldSize_ * normalOffset * (0.5f + slope));

        const Vec4 clip = lightViewProjection_ * Vec4(samplePosition, 1.0f);
        if (clip.w <= 0.0f) return 1.0f;

        const float invW = 1.0f / clip.w;
        const float u = clip.x * invW * 0.5f + 0.5f;
        const float v = 0.5f - clip.y * invW * 0.5f;
        const float z = clip.z * invW * 0.5f + 0.5f;

        // Anything outside the map is treated as lit. The map only has to cover
        // the casters, so a large ground plane can extend well past it.
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
