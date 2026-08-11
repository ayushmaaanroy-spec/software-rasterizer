// Minimal PNG encoder: CRC-32, Adler-32, and a fixed-Huffman DEFLATE compressor,
// all written here so the project keeps its no-dependencies rule while still
// producing an image format that renders inline on GitHub.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sr {

// `rgb` is 8-bit RGB, three bytes per pixel, top-down, with no row padding.
// Returns false only on invalid arguments or an I/O failure.
bool writePng(const std::string& path, int width, int height, const std::uint8_t* rgb);

// Exposed for testing: the raw PNG byte stream that writePng would put on disk.
[[nodiscard]] std::vector<std::uint8_t> encodePng(int width, int height, const std::uint8_t* rgb);

}  // namespace sr
