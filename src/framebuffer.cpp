#include "sr/framebuffer.hpp"

#include <fstream>
#include <limits>

#include "sr/png.hpp"

namespace sr {
namespace {

// BMP is little-endian whatever the host is.
void put16(std::uint8_t* out, std::uint16_t v) {
    out[0] = static_cast<std::uint8_t>(v & 0xFF);
    out[1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
}

void put32(std::uint8_t* out, std::uint32_t v) {
    out[0] = static_cast<std::uint8_t>(v & 0xFF);
    out[1] = static_cast<std::uint8_t>((v >> 8) & 0xFF);
    out[2] = static_cast<std::uint8_t>((v >> 16) & 0xFF);
    out[3] = static_cast<std::uint8_t>((v >> 24) & 0xFF);
}

std::uint8_t encodeChannel(float linear) {
    return static_cast<std::uint8_t>(clampf(linearToSrgb(linear), 0.0f, 1.0f) * 255.0f + 0.5f);
}

}  // namespace

float linearToSrgb(float linear) noexcept {
    if (linear <= 0.0031308f) return linear * 12.92f;
    return 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
}

float srgbToLinear(float encoded) noexcept {
    if (encoded <= 0.04045f) return encoded / 12.92f;
    return std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

Framebuffer::Framebuffer(int width, int height, DepthOnly depthOnly)
    : width_(width > 0 ? width : 1),
      height_(height > 0 ? height : 1),
      color_(depthOnly == DepthOnly::Yes ? 0 : static_cast<std::size_t>(width_) * height_),
      depth_(static_cast<std::size_t>(width_) * height_, 1.0f) {}

void Framebuffer::clearColor(const Vec3& color) noexcept {
    for (Vec3& c : color_) c = color;
}

void Framebuffer::clearDepth(float depth) noexcept {
    for (float& d : depth_) d = depth;
}

void Framebuffer::clear(const Vec3& color, float depth) noexcept {
    clearColor(color);
    clearDepth(depth);
}

Framebuffer Framebuffer::downsample(int factor) const {
    if (factor <= 1 || !hasColor()) return *this;

    Framebuffer out(width_ / factor, height_ / factor);
    const float inv = 1.0f / static_cast<float>(factor * factor);

    for (int y = 0; y < out.height_; ++y) {
        for (int x = 0; x < out.width_; ++x) {
            Vec3 sum;
            float depth = 1.0f;
            for (int sy = 0; sy < factor; ++sy) {
                for (int sx = 0; sx < factor; ++sx) {
                    sum += colorAt(x * factor + sx, y * factor + sy);
                    depth = std::min(depth, depthAt(x * factor + sx, y * factor + sy));
                }
            }
            out.colorAt(x, y) = sum * inv;
            out.depthAt(x, y) = depth;
        }
    }
    return out;
}

void Framebuffer::visualizeDepth() {
    if (!hasColor()) return;

    float lo = std::numeric_limits<float>::max();
    float hi = 0.0f;
    for (float d : depth_) {
        if (d >= 1.0f) continue;  // background
        lo = std::min(lo, d);
        hi = std::max(hi, d);
    }
    const float range = (hi > lo) ? (hi - lo) : 1.0f;

    for (std::size_t i = 0; i < depth_.size(); ++i) {
        if (depth_[i] >= 1.0f) {
            color_[i] = Vec3(0.0f);
            continue;
        }
        // Near geometry reads bright, far geometry dark.
        const float t = 1.0f - (depth_[i] - lo) / range;
        color_[i] = Vec3(t * t);
    }
}

std::vector<std::uint8_t> Framebuffer::encodeRgb8() const {
    if (!hasColor()) return {};

    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width_) * height_ * 3);
    std::size_t cursor = 0;
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec3& c = colorAt(x, y);
            rgb[cursor++] = encodeChannel(c.x);
            rgb[cursor++] = encodeChannel(c.y);
            rgb[cursor++] = encodeChannel(c.z);
        }
    }
    return rgb;
}

bool Framebuffer::writePNG(const std::string& path) const {
    const std::vector<std::uint8_t> rgb = encodeRgb8();
    return writePng(path, width_, height_, rgb.data());
}

bool Framebuffer::writeBMP(const std::string& path) const {
    if (!hasColor()) return false;

    const int rowBytes = width_ * 3;
    const int padding = (4 - (rowBytes % 4)) % 4;  // rows align to 4 bytes
    const std::uint32_t imageBytes = static_cast<std::uint32_t>((rowBytes + padding) * height_);
    constexpr std::uint32_t kHeaderBytes = 14 + 40;

    // Zero-filled, so the reserved fields and the row padding need no writes.
    std::vector<std::uint8_t> bytes(kHeaderBytes + imageBytes, 0);
    std::uint8_t* const header = bytes.data();

    header[0] = 'B';
    header[1] = 'M';
    put32(header + 2, kHeaderBytes + imageBytes);
    put32(header + 10, kHeaderBytes);       // offset to pixel data

    put32(header + 14, 40);                 // BITMAPINFOHEADER size
    put32(header + 18, static_cast<std::uint32_t>(width_));
    put32(header + 22, static_cast<std::uint32_t>(height_));
    put16(header + 26, 1);                  // color planes
    put16(header + 28, 24);                 // bits per pixel
    put32(header + 30, 0);                  // BI_RGB, no compression
    put32(header + 34, imageBytes);
    put32(header + 38, 2835);               // ~72 DPI, in pixels per metre
    put32(header + 42, 2835);

    // Rows are stored bottom-up.
    std::size_t cursor = kHeaderBytes;
    for (int y = height_ - 1; y >= 0; --y) {
        for (int x = 0; x < width_; ++x) {
            const Vec3& c = colorAt(x, y);
            bytes[cursor++] = encodeChannel(c.z);  // BMP stores BGR
            bytes[cursor++] = encodeChannel(c.y);
            bytes[cursor++] = encodeChannel(c.x);
        }
        cursor += static_cast<std::size_t>(padding);
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(file);
}

bool Framebuffer::writePPM(const std::string& path) const {
    if (!hasColor()) return false;

    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    file << "P6\n" << width_ << " " << height_ << "\n255\n";
    std::vector<std::uint8_t> row(static_cast<std::size_t>(width_) * 3);
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const Vec3& c = colorAt(x, y);
            row[x * 3 + 0] = encodeChannel(c.x);
            row[x * 3 + 1] = encodeChannel(c.y);
            row[x * 3 + 2] = encodeChannel(c.z);
        }
        file.write(reinterpret_cast<const char*>(row.data()),
                   static_cast<std::streamsize>(row.size()));
    }
    return static_cast<bool>(file);
}

}  // namespace sr
