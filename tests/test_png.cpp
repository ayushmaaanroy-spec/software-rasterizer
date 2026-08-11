// Structural checks on the PNG encoder. A malformed stream is caught here; that
// real decoders accept the output is verified separately against an external
// decoder, since this project deliberately has nothing to decode it with.
#include <cstdint>
#include <string>
#include <vector>

#include "check.hpp"
#include "sr/framebuffer.hpp"
#include "sr/png.hpp"

using namespace sr;

namespace {

std::uint32_t readBigEndian32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return (static_cast<std::uint32_t>(bytes[offset]) << 24) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
           static_cast<std::uint32_t>(bytes[offset + 3]);
}

std::uint32_t crc32(const std::uint8_t* data, std::size_t count) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < count; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) crc = (crc & 1u) ? (0xEDB88320u ^ (crc >> 1)) : (crc >> 1);
    }
    return crc ^ 0xFFFFFFFFu;
}

// A gradient with a few hard edges: compressible, but not trivially so.
std::vector<std::uint8_t> testImage(int width, int height) {
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(width) * height * 3);
    std::size_t cursor = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            rgb[cursor++] = static_cast<std::uint8_t>(x * 255 / (width - 1));
            rgb[cursor++] = static_cast<std::uint8_t>(y * 255 / (height - 1));
            rgb[cursor++] = static_cast<std::uint8_t>(((x / 8) + (y / 8)) % 2 ? 240 : 16);
        }
    }
    return rgb;
}

}  // namespace

TEST(png_stream_is_structurally_valid) {
    constexpr int kWidth = 61, kHeight = 37;  // deliberately not round numbers
    const std::vector<std::uint8_t> rgb = testImage(kWidth, kHeight);
    const std::vector<std::uint8_t> png = encodePng(kWidth, kHeight, rgb.data());

    CHECK(png.size() > 8);

    static constexpr std::uint8_t kSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    bool signatureOk = true;
    for (std::size_t i = 0; i < 8; ++i)
        if (png[i] != kSignature[i]) signatureOk = false;
    CHECK(signatureOk);

    // Walk the chunk list, validating every length and CRC as we go.
    std::string order;
    bool crcsOk = true;
    std::size_t offset = 8;
    while (offset + 12 <= png.size()) {
        const std::uint32_t length = readBigEndian32(png, offset);
        const std::size_t typeOffset = offset + 4;
        if (typeOffset + 4 + length + 4 > png.size()) {
            crcsOk = false;
            break;
        }
        order.append(reinterpret_cast<const char*>(&png[typeOffset]), 4);

        const std::uint32_t expected = crc32(&png[typeOffset], 4 + length);
        if (readBigEndian32(png, typeOffset + 4 + length) != expected) crcsOk = false;

        offset = typeOffset + 4 + length + 4;
    }
    CHECK(crcsOk);
    CHECK(offset == png.size());  // no trailing garbage
    CHECK(order == "IHDRIDATIEND");

    // IHDR: 8-bit truecolour RGB, no interlacing, dimensions as requested.
    CHECK(readBigEndian32(png, 16) == static_cast<std::uint32_t>(kWidth));
    CHECK(readBigEndian32(png, 20) == static_cast<std::uint32_t>(kHeight));
    CHECK(png[24] == 8);  // bit depth
    CHECK(png[25] == 2);  // colour type
    CHECK(png[26] == 0);  // compression method
    CHECK(png[27] == 0);  // filter method
    CHECK(png[28] == 0);  // interlace method
}

TEST(png_zlib_wrapper_is_well_formed) {
    constexpr int kWidth = 40, kHeight = 40;
    const std::vector<std::uint8_t> rgb = testImage(kWidth, kHeight);
    const std::vector<std::uint8_t> png = encodePng(kWidth, kHeight, rgb.data());

    // IDAT begins right after the 8-byte signature and the 25-byte IHDR chunk.
    const std::size_t idatLength = readBigEndian32(png, 33);
    const std::size_t idatData = 41;
    CHECK(idatLength >= 6);

    // zlib header: deflate, 32K window, no preset dictionary, and the two bytes
    // together must be a multiple of 31.
    const std::uint8_t cmf = png[idatData];
    const std::uint8_t flg = png[idatData + 1];
    CHECK((cmf & 0x0F) == 8);
    CHECK((cmf >> 4) <= 7);
    CHECK((flg & 0x20) == 0);
    CHECK(((static_cast<unsigned>(cmf) << 8 | flg) % 31u) == 0);

    // First deflate block: BFINAL set, BTYPE 01 (fixed Huffman).
    const std::uint8_t firstBlock = png[idatData + 2];
    CHECK((firstBlock & 0x01) == 1);
    CHECK(((firstBlock >> 1) & 0x03) == 1);

    // Adler-32 of the filtered scanlines, recomputed independently.
    const std::size_t stride = static_cast<std::size_t>(kWidth) * 3;
    std::uint32_t a = 1, b = 0;
    for (int y = 0; y < kHeight; ++y) {
        const std::uint8_t* row = rgb.data() + static_cast<std::size_t>(y) * stride;
        const std::uint8_t* above = (y == 0) ? nullptr : row - stride;
        a = (a + 2u) % 65521u;  // the Up filter byte
        b = (b + a) % 65521u;
        for (std::size_t i = 0; i < stride; ++i) {
            const std::uint8_t filtered =
                static_cast<std::uint8_t>(row[i] - (above != nullptr ? above[i] : 0));
            a = (a + filtered) % 65521u;
            b = (b + a) % 65521u;
        }
    }
    CHECK(readBigEndian32(png, idatData + idatLength - 4) == ((b << 16) | a));
}

TEST(png_actually_compresses) {
    constexpr int kWidth = 256, kHeight = 256;
    const std::vector<std::uint8_t> rgb = testImage(kWidth, kHeight);
    const std::vector<std::uint8_t> png = encodePng(kWidth, kHeight, rgb.data());

    // Stored deflate blocks would come out larger than the input; this asserts
    // the LZ77 and Huffman paths are doing real work.
    CHECK(png.size() < rgb.size() / 2);
    CHECK(png.size() > 100);
}

TEST(png_rejects_bad_arguments) {
    const std::vector<std::uint8_t> rgb = testImage(8, 8);
    CHECK(encodePng(0, 8, rgb.data()).empty());
    CHECK(encodePng(8, -1, rgb.data()).empty());
    CHECK(encodePng(8, 8, nullptr).empty());
}

TEST(framebuffer_rgb8_matches_the_srgb_curve) {
    Framebuffer fb(2, 1);
    fb.colorAt(0, 0) = Vec3{0.0f, 0.5f, 1.0f};
    fb.colorAt(1, 0) = Vec3{1.0f, 1.0f, 1.0f};

    const std::vector<std::uint8_t> rgb = fb.encodeRgb8();
    CHECK(rgb.size() == 6);
    CHECK(rgb[0] == 0);
    // Linear 0.5 encodes to about 188, not 128 -- that is the point of the curve.
    CHECK(rgb[1] > 180 && rgb[1] < 195);
    CHECK(rgb[2] == 255);
    CHECK(rgb[3] == 255 && rgb[4] == 255 && rgb[5] == 255);
}
