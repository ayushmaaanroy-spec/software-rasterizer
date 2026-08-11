#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sr {

// rgb is 8-bit RGB, top-down, no row padding.
bool writePng(const std::string& path, int width, int height, const std::uint8_t* rgb);

// Same bytes, without touching the disk. Used by the tests.
[[nodiscard]] std::vector<std::uint8_t> encodePng(int width, int height, const std::uint8_t* rgb);

}  // namespace sr
