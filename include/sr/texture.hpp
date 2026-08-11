#pragma once

#include <vector>

#include "sr/math.hpp"

namespace sr {

enum class WrapMode { Repeat, Clamp };

// Texels are linear RGB, matching the framebuffer.
class Texture {
public:
    Texture() = default;
    Texture(int width, int height);

    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] bool empty() const noexcept { return texels_.empty(); }

    [[nodiscard]] Vec3& texelAt(int x, int y) noexcept {
        return texels_[static_cast<std::size_t>(y) * width_ + x];
    }
    [[nodiscard]] const Vec3& texelAt(int x, int y) const noexcept {
        return texels_[static_cast<std::size_t>(y) * width_ + x];
    }

    [[nodiscard]] Vec3 sampleNearest(float u, float v) const noexcept;
    [[nodiscard]] Vec3 sampleBilinear(float u, float v) const noexcept;

    void setWrap(WrapMode mode) noexcept { wrap_ = mode; }

    [[nodiscard]] static Texture checker(int size, int cells, const Vec3& a, const Vec3& b);
    [[nodiscard]] static Texture grid(int size, int cells, const Vec3& line, const Vec3& fill,
                                      int lineWidth = 2);
    [[nodiscard]] static Texture noise(int size, const Vec3& a, const Vec3& b, unsigned seed = 1337);

private:
    [[nodiscard]] int wrapCoord(int c, int limit) const noexcept;

    int width_ = 0;
    int height_ = 0;
    WrapMode wrap_ = WrapMode::Repeat;
    std::vector<Vec3> texels_;
};

}  // namespace sr
