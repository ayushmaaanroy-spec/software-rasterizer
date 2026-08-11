// Indexed triangle meshes. Front faces are CCW seen from outside.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sr/math.hpp"

namespace sr {

struct Vertex {
    Vec3 position;
    Vec3 normal{0.0f, 1.0f, 0.0f};
    Vec2 uv;
    Vec3 color{1.0f, 1.0f, 1.0f};
};

struct Bounds {
    Vec3 min{0.0f, 0.0f, 0.0f};
    Vec3 max{0.0f, 0.0f, 0.0f};

    [[nodiscard]] Vec3 center() const noexcept { return (min + max) * 0.5f; }
    [[nodiscard]] Vec3 extent() const noexcept { return max - min; }
    [[nodiscard]] float radius() const noexcept { return length(extent()) * 0.5f; }
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    [[nodiscard]] std::size_t triangleCount() const noexcept { return indices.size() / 3; }

    [[nodiscard]] Bounds bounds() const noexcept;

    void recomputeNormals() noexcept;
    void flipWinding() noexcept;
    void setColor(const Vec3& color) noexcept;
    void transform(const Mat4& matrix) noexcept;
    // Center on the origin and scale the largest dimension to targetSize.
    void normalizeToUnitSize(float targetSize = 2.0f) noexcept;
    void append(const Mesh& other);

    [[nodiscard]] static Mesh cube(float size = 1.0f);
    [[nodiscard]] static Mesh plane(float size, int subdivisions = 1);
    [[nodiscard]] static Mesh uvSphere(float radius = 1.0f, int stacks = 24, int slices = 48);
    [[nodiscard]] static Mesh torus(float majorRadius = 1.0f, float minorRadius = 0.35f,
                                    int majorSegments = 48, int minorSegments = 24);

    // v/vt/vn, with polygonal faces fan-triangulated. nullopt on failure, with a
    // reason in error if given.
    [[nodiscard]] static std::optional<Mesh> loadObj(const std::string& path,
                                                     std::string* error = nullptr);
};

}  // namespace sr
