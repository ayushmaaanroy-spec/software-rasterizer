#include "sr/texture.hpp"

namespace sr {
namespace {

// Deterministic hash-based value noise: no <random>, no allocation, same result
// on every platform.
float hashNoise(int x, int y, unsigned seed) {
    unsigned h = seed;
    h ^= static_cast<unsigned>(x) * 0x9E3779B1u;
    h ^= static_cast<unsigned>(y) * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0x2545F491u;
    h ^= h >> 13;
    return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFF);
}

float smoothNoise(float x, float y, unsigned seed) {
    const int ix = static_cast<int>(std::floor(x));
    const int iy = static_cast<int>(std::floor(y));
    const float fx = x - static_cast<float>(ix);
    const float fy = y - static_cast<float>(iy);
    // Smoothstep weights give C1 continuity between lattice points.
    const float sx = fx * fx * (3.0f - 2.0f * fx);
    const float sy = fy * fy * (3.0f - 2.0f * fy);

    const float n00 = hashNoise(ix, iy, seed);
    const float n10 = hashNoise(ix + 1, iy, seed);
    const float n01 = hashNoise(ix, iy + 1, seed);
    const float n11 = hashNoise(ix + 1, iy + 1, seed);

    const float a = n00 + (n10 - n00) * sx;
    const float b = n01 + (n11 - n01) * sx;
    return a + (b - a) * sy;
}

}  // namespace

Texture::Texture(int width, int height)
    : width_(width > 0 ? width : 1),
      height_(height > 0 ? height : 1),
      texels_(static_cast<std::size_t>(width_) * height_) {}

int Texture::wrapCoord(int c, int limit) const noexcept {
    if (wrap_ == WrapMode::Clamp) return c < 0 ? 0 : (c >= limit ? limit - 1 : c);
    c %= limit;
    return c < 0 ? c + limit : c;
}

Vec3 Texture::sampleNearest(float u, float v) const noexcept {
    if (texels_.empty()) return Vec3(1.0f);
    const int x = wrapCoord(static_cast<int>(std::floor(u * static_cast<float>(width_))), width_);
    const int y = wrapCoord(static_cast<int>(std::floor(v * static_cast<float>(height_))), height_);
    return texelAt(x, y);
}

Vec3 Texture::sampleBilinear(float u, float v) const noexcept {
    if (texels_.empty()) return Vec3(1.0f);

    // Sample at texel centers, hence the -0.5 shift.
    const float fx = u * static_cast<float>(width_) - 0.5f;
    const float fy = v * static_cast<float>(height_) - 0.5f;
    const int x0 = static_cast<int>(std::floor(fx));
    const int y0 = static_cast<int>(std::floor(fy));
    const float tx = fx - static_cast<float>(x0);
    const float ty = fy - static_cast<float>(y0);

    const int xa = wrapCoord(x0, width_), xb = wrapCoord(x0 + 1, width_);
    const int ya = wrapCoord(y0, height_), yb = wrapCoord(y0 + 1, height_);

    const Vec3 top = lerp(texelAt(xa, ya), texelAt(xb, ya), tx);
    const Vec3 bottom = lerp(texelAt(xa, yb), texelAt(xb, yb), tx);
    return lerp(top, bottom, ty);
}

Texture Texture::checker(int size, int cells, const Vec3& a, const Vec3& b) {
    Texture t(size, size);
    const int cellSize = std::max(1, size / std::max(1, cells));
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const bool odd = ((x / cellSize) + (y / cellSize)) & 1;
            t.texelAt(x, y) = odd ? b : a;
        }
    }
    return t;
}

Texture Texture::grid(int size, int cells, const Vec3& line, const Vec3& fill, int lineWidth) {
    Texture t(size, size);
    const int cellSize = std::max(1, size / std::max(1, cells));
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const bool onLine = (x % cellSize) < lineWidth || (y % cellSize) < lineWidth;
            t.texelAt(x, y) = onLine ? line : fill;
        }
    }
    return t;
}

Texture Texture::noise(int size, const Vec3& a, const Vec3& b, unsigned seed) {
    Texture t(size, size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            // Four octaves of value noise, each half the amplitude of the last.
            float amplitude = 0.5f;
            float frequency = 8.0f / static_cast<float>(size);
            float sum = 0.0f;
            for (int octave = 0; octave < 4; ++octave) {
                sum += amplitude * smoothNoise(static_cast<float>(x) * frequency,
                                               static_cast<float>(y) * frequency, seed + octave);
                amplitude *= 0.5f;
                frequency *= 2.0f;
            }
            t.texelAt(x, y) = lerp(a, b, clampf(sum, 0.0f, 1.0f));
        }
    }
    return t;
}

}  // namespace sr
