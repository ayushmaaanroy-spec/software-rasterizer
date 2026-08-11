#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sr/math.hpp"

namespace sr {

// A shadow map never reads its color buffer, and at 2048 squared that is 50 MB
// of nothing.
enum class DepthOnly { No, Yes };

// Color is linear float RGB; sRGB is applied once, on write. Depth is window z
// in [0, 1].
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

    // How supersampling resolves to the final image.
    [[nodiscard]] Framebuffer downsample(int factor) const;

    // Replaces color with the depth buffer, stretched over the range geometry
    // actually covers.
    void visualizeDepth();

    // sRGB-encoded 8-bit RGB, top-down and unpadded.
    [[nodiscard]] std::vector<std::uint8_t> encodeRgb8() const;

    bool writePNG(const std::string& path) const;
    bool writeBMP(const std::string& path) const;
    bool writePPM(const std::string& path) const;

private:
    int width_;
    int height_;
    std::vector<Vec3> color_;
    std::vector<float> depth_;
};

[[nodiscard]] float linearToSrgb(float linear) noexcept;
[[nodiscard]] float srgbToLinear(float encoded) noexcept;

}  // namespace sr
