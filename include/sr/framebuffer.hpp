// Colour + depth render target, and the image writers used to get pixels to disk.
#pragma once

#include <cstdint>
#include <string>
#include <vector>


#include "sr/math.hpp"

namespace sr {

// A shadow map never reads its colour buffer, and at 2048 squared that buffer is
// 50 MB of nothing. Depth-only targets skip the allocation entirely.
enum class DepthOnly { No, Yes };

// Colour is stored as linear-space float RGB; the sRGB transfer curve is applied
// once, on write. Depth is window-space z in [0, 1], where 0 is the near plane.
class Framebuffer {
public:
    Framebuffer(int width, int height, DepthOnly depthOnly = DepthOnly::No);

    // False for depth-only targets, on which colorAt() must not be called.
    [[nodiscard]] bool hasColor() const noexcept { return !color_.empty(); }

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] float aspect() const noexcept {
        return static_cast<float>(width_) / static_cast<float>(height_);
    }

    void clearColor(const Vec3& color) noexcept;
    void clearDepth(float depth = 1.0f) noexcept;
    void clear(const Vec3& color, float depth = 1.0f) noexcept;

    [[nodiscard]] Vec3& colorAt(int x, int y) noexcept {
        return color_[static_cast<std::size_t>(y) * width_ + x];
    }
    [[nodiscard]] const Vec3& colorAt(int x, int y) const noexcept {
        return color_[static_cast<std::size_t>(y) * width_ + x];
    }
    [[nodiscard]] float& depthAt(int x, int y) noexcept {
        return depth_[static_cast<std::size_t>(y) * width_ + x];
    }
    [[nodiscard]] float depthAt(int x, int y) const noexcept {
        return depth_[static_cast<std::size_t>(y) * width_ + x];
    }

    // Box-filtered downsample by an integer factor; this is how supersampling
    // (SSAA) resolves to the final image.
    [[nodiscard]] Framebuffer downsample(int factor) const;

    // Replaces colour with a visualisation of the depth buffer, contrast
    // stretched across the range actually covered by geometry.
    void visualizeDepth();

    // sRGB-encoded 8-bit RGB, three bytes per pixel, top-down and unpadded.
    [[nodiscard]] std::vector<std::uint8_t> encodeRgb8() const;

    // All three writers gamma-encode on the way out. PNG is the default because
    // it is lossless and renders inline on GitHub; BMP is 24-bit BGR bottom-up.
    bool writePNG(const std::string& path) const;
    bool writeBMP(const std::string& path) const;
    bool writePPM(const std::string& path) const;

private:
    int width_;
    int height_;
    std::vector<Vec3> color_;
    std::vector<float> depth_;
};

// Linear -> sRGB transfer function (the piecewise-exact one, not the 2.2 approximation).
[[nodiscard]] float linearToSrgb(float linear) noexcept;
[[nodiscard]] float srgbToLinear(float encoded) noexcept;

}  // namespace sr
