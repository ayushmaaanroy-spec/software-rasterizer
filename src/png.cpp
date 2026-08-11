// PNG writer: CRC-32, Adler-32 and a fixed-Huffman DEFLATE compressor, so image
// output needs no dependency either.
#include "sr/png.hpp"

#include <algorithm>
#include <array>
#include <fstream>

namespace sr {
namespace {

std::uint32_t crc32Update(std::uint32_t crc, const std::uint8_t* data, std::size_t count) {
    static const std::array<std::uint32_t, 256> kTable = [] {
        std::array<std::uint32_t, 256> table{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int bit = 0; bit < 8; ++bit) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        return table;
    }();

    for (std::size_t i = 0; i < count; ++i) crc = kTable[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

std::uint32_t adler32(const std::uint8_t* data, std::size_t count) {
    std::uint32_t a = 1, b = 0;
    for (std::size_t i = 0; i < count; ++i) {
        a = (a + data[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void appendBigEndian32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<std::uint8_t>(v & 0xFF));
}

// DEFLATE packs header fields LSB first but Huffman codes MSB first. Two orders
// in the same stream, hence two methods.
class BitWriter {
public:
    explicit BitWriter(std::vector<std::uint8_t>& out) noexcept : out_(out) {}

    void writeBits(std::uint32_t value, int count) {
        for (int i = 0; i < count; ++i) pushBit((value >> i) & 1u);
    }

    void writeCode(std::uint32_t code, int count) {
        for (int i = count - 1; i >= 0; --i) pushBit((code >> i) & 1u);
    }

    void alignToByte() {
        if (bitCount_ != 0) flushByte();
    }

private:
    void pushBit(std::uint32_t bit) {
        buffer_ |= bit << bitCount_;
        if (++bitCount_ == 8) flushByte();
    }
    void flushByte() {
        out_.push_back(static_cast<std::uint8_t>(buffer_));
        buffer_ = 0;
        bitCount_ = 0;
    }

    std::vector<std::uint8_t>& out_;
    std::uint32_t buffer_ = 0;
    int bitCount_ = 0;
};

// The fixed literal/length code from RFC 1951 section 3.2.6.
void writeLiteral(BitWriter& bits, int symbol) {
    if (symbol <= 143) {
        bits.writeCode(0x30u + static_cast<std::uint32_t>(symbol), 8);
    } else if (symbol <= 255) {
        bits.writeCode(0x190u + static_cast<std::uint32_t>(symbol - 144), 9);
    } else if (symbol <= 279) {
        bits.writeCode(static_cast<std::uint32_t>(symbol - 256), 7);
    } else {
        bits.writeCode(0xC0u + static_cast<std::uint32_t>(symbol - 280), 8);
    }
}

struct RangeCode {
    int symbol;
    int extraBits;
    int base;
};

// RFC 1951 section 3.2.5.
constexpr RangeCode kLengthCodes[] = {
    {257, 0, 3},    {258, 0, 4},    {259, 0, 5},    {260, 0, 6},   {261, 0, 7},   {262, 0, 8},
    {263, 0, 9},    {264, 0, 10},   {265, 1, 11},   {266, 1, 13},  {267, 1, 15},  {268, 1, 17},
    {269, 2, 19},   {270, 2, 23},   {271, 2, 27},   {272, 2, 31},  {273, 3, 35},  {274, 3, 43},
    {275, 3, 51},   {276, 3, 59},   {277, 4, 67},   {278, 4, 83},  {279, 4, 99},  {280, 4, 115},
    {281, 5, 131},  {282, 5, 163},  {283, 5, 195},  {284, 5, 227}, {285, 0, 258}};

constexpr RangeCode kDistanceCodes[] = {
    {0, 0, 1},        {1, 0, 2},        {2, 0, 3},        {3, 0, 4},        {4, 1, 5},
    {5, 1, 7},        {6, 2, 9},        {7, 2, 13},       {8, 3, 17},       {9, 3, 25},
    {10, 4, 33},      {11, 4, 49},      {12, 5, 65},      {13, 5, 97},      {14, 6, 129},
    {15, 6, 193},     {16, 7, 257},     {17, 7, 385},     {18, 8, 513},     {19, 8, 769},
    {20, 9, 1025},    {21, 9, 1537},    {22, 10, 2049},   {23, 10, 3073},   {24, 11, 4097},
    {25, 11, 6145},   {26, 12, 8193},   {27, 12, 12289},  {28, 13, 16385},  {29, 13, 24577}};

template <std::size_t N>
const RangeCode& selectRange(const RangeCode (&table)[N], int value) {
    // Tables are ascending, so the last entry whose base fits is the right one.
    std::size_t index = N - 1;
    while (index > 0 && table[index].base > value) --index;
    return table[index];
}

void writeLength(BitWriter& bits, int length) {
    const RangeCode& range = selectRange(kLengthCodes, length);
    writeLiteral(bits, range.symbol);
    if (range.extraBits > 0)
        bits.writeBits(static_cast<std::uint32_t>(length - range.base), range.extraBits);
}

void writeDistance(BitWriter& bits, int distance) {
    const RangeCode& range = selectRange(kDistanceCodes, distance);
    // Distances use a fixed 5-bit code, not the literal/length alphabet.
    bits.writeCode(static_cast<std::uint32_t>(range.symbol), 5);
    if (range.extraBits > 0)
        bits.writeBits(static_cast<std::uint32_t>(distance - range.base), range.extraBits);
}

constexpr int kWindowSize = 32768;
constexpr int kMinMatch = 3;
constexpr int kMaxMatch = 258;
constexpr int kHashBits = 15;
constexpr int kHashSize = 1 << kHashBits;
constexpr int kMaxChainLength = 32;  // caps the search; trades ratio for speed

std::uint32_t hash3(const std::uint8_t* p) {
    const std::uint32_t key = (static_cast<std::uint32_t>(p[0]) << 16) |
                              (static_cast<std::uint32_t>(p[1]) << 8) |
                              static_cast<std::uint32_t>(p[2]);
    return (key * 2654435761u) >> (32 - kHashBits);
}

// One fixed-Huffman block, greedy LZ77 over a hash chain.
void deflateFixed(const std::vector<std::uint8_t>& src, std::vector<std::uint8_t>& out) {
    BitWriter bits(out);
    bits.writeBits(1, 1);  // BFINAL, this is the only block
    bits.writeBits(1, 2);  // BTYPE 01, fixed Huffman

    const int size = static_cast<int>(src.size());
    std::vector<int> head(kHashSize, -1);
    std::vector<int> prev(src.size(), -1);

    auto insert = [&](int position) {
        const std::uint32_t h = hash3(&src[static_cast<std::size_t>(position)]);
        prev[static_cast<std::size_t>(position)] = head[h];
        head[h] = position;
    };

    int position = 0;
    while (position < size) {
        int bestLength = 0;
        int bestDistance = 0;

        if (position + kMinMatch <= size) {
            const int maxLength = std::min(kMaxMatch, size - position);
            int candidate = head[hash3(&src[static_cast<std::size_t>(position)])];

            for (int chain = 0; candidate >= 0 && chain < kMaxChainLength; ++chain) {
                const int distance = position - candidate;
                if (distance > kWindowSize) break;  // chain is ordered, so are the rest

                int length = 0;
                while (length < maxLength && src[static_cast<std::size_t>(candidate + length)] ==
                                                 src[static_cast<std::size_t>(position + length)]) {
                    ++length;
                }
                if (length > bestLength) {
                    bestLength = length;
                    bestDistance = distance;
                    if (length == maxLength) break;
                }
                candidate = prev[static_cast<std::size_t>(candidate)];
            }
            insert(position);
        }

        if (bestLength >= kMinMatch) {
            writeLength(bits, bestLength);
            writeDistance(bits, bestDistance);
            // Register what the match covered so later matches can find it.
            for (int i = 1; i < bestLength; ++i)
                if (position + i + kMinMatch <= size) insert(position + i);
            position += bestLength;
        } else {
            writeLiteral(bits, src[static_cast<std::size_t>(position)]);
            ++position;
        }
    }

    writeLiteral(bits, 256);  // end-of-block
    bits.alignToByte();
}

std::vector<std::uint8_t> zlibCompress(const std::vector<std::uint8_t>& raw) {
    std::vector<std::uint8_t> out;
    // CM 8 (deflate), CINFO 7 (32K window), FCHECK making the pair a multiple of 31.
    out.push_back(0x78);
    out.push_back(0x01);
    deflateFixed(raw, out);
    appendBigEndian32(out, adler32(raw.data(), raw.size()));
    return out;
}

void appendChunk(std::vector<std::uint8_t>& out, const char (&type)[5], const std::uint8_t* data,
                 std::size_t size) {
    appendBigEndian32(out, static_cast<std::uint32_t>(size));
    const std::size_t crcBegin = out.size();
    out.insert(out.end(), type, type + 4);
    if (size > 0) out.insert(out.end(), data, data + size);
    // CRC covers the type and data, not the length.
    const std::uint32_t crc =
        crc32Update(0xFFFFFFFFu, out.data() + crcBegin, out.size() - crcBegin) ^ 0xFFFFFFFFu;
    appendBigEndian32(out, crc);
}

}  // namespace

std::vector<std::uint8_t> encodePng(int width, int height, const std::uint8_t* rgb) {
    std::vector<std::uint8_t> png;
    if (width <= 0 || height <= 0 || rgb == nullptr) return png;

    const std::size_t stride = static_cast<std::size_t>(width) * 3;

    // Each scanline is prefixed with its filter type. Up (2) subtracts the pixel
    // above, turning the smooth gradients a renderer makes into runs of
    // near-zero bytes that LZ77 eats cheaply.
    std::vector<std::uint8_t> raw;
    raw.reserve((stride + 1) * static_cast<std::size_t>(height));
    for (int y = 0; y < height; ++y) {
        raw.push_back(2);
        const std::uint8_t* row = rgb + static_cast<std::size_t>(y) * stride;
        // Row 0 filters against an implicit row of zeros, per the spec.
        const std::uint8_t* above = (y == 0) ? nullptr : row - stride;
        for (std::size_t i = 0; i < stride; ++i) {
            const std::uint8_t previous = above != nullptr ? above[i] : 0;
            raw.push_back(static_cast<std::uint8_t>(row[i] - previous));
        }
    }

    static constexpr std::uint8_t kSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    png.insert(png.end(), kSignature, kSignature + 8);

    std::vector<std::uint8_t> ihdr;
    appendBigEndian32(ihdr, static_cast<std::uint32_t>(width));
    appendBigEndian32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // color type 2, truecolor RGB
    ihdr.push_back(0);  // compression method: deflate
    ihdr.push_back(0);  // filter method: adaptive
    ihdr.push_back(0);  // interlace: none
    appendChunk(png, "IHDR", ihdr.data(), ihdr.size());

    const std::vector<std::uint8_t> compressed = zlibCompress(raw);
    appendChunk(png, "IDAT", compressed.data(), compressed.size());
    appendChunk(png, "IEND", nullptr, 0);

    return png;
}

bool writePng(const std::string& path, int width, int height, const std::uint8_t* rgb) {
    const std::vector<std::uint8_t> png = encodePng(width, height, rgb);
    if (png.empty()) return false;

    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return static_cast<bool>(file);
}

}  // namespace sr
