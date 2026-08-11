#include "sr/mesh.hpp"

#include <fstream>
#include <sstream>
#include <unordered_map>

namespace sr {
namespace {

// Two triangles of the (i, j) quad, wound CCW seen from outside. Shared by every
// parametric surface below.
void emitQuad(std::vector<std::uint32_t>& indices, int i, int j, int stride) {
    const std::uint32_t a = static_cast<std::uint32_t>(i * stride + j);
    const std::uint32_t b = static_cast<std::uint32_t>(i * stride + j + 1);
    const std::uint32_t c = static_cast<std::uint32_t>((i + 1) * stride + j + 1);
    const std::uint32_t d = static_cast<std::uint32_t>((i + 1) * stride + j);

    indices.insert(indices.end(), {a, b, c});
    indices.insert(indices.end(), {a, c, d});
}

}  // namespace

Bounds Mesh::bounds() const noexcept {
    if (vertices.empty()) return {};

    Bounds b{vertices[0].position, vertices[0].position};
    for (const Vertex& v : vertices) {
        b.min = minv(b.min, v.position);
        b.max = maxv(b.max, v.position);
    }
    return b;
}

void Mesh::recomputeNormals() noexcept {
    std::vector<Vec3> accumulated(vertices.size(), Vec3(0.0f));

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        const Vec3& a = vertices[indices[i + 0]].position;
        const Vec3& b = vertices[indices[i + 1]].position;
        const Vec3& c = vertices[indices[i + 2]].position;

        const Vec3 edge1 = b - a;
        const Vec3 edge2 = c - a;
        // Left unnormalized, which weights each face by twice its area.
        const Vec3 faceNormal = cross(edge1, edge2);

        // A sliver's cross product is mostly round-off pointing anywhere, so drop
        // it rather than average it in. Sphere caps and fan-triangulated OBJ
        // faces both produce these. Scale-relative, since |e1 x e2| = |e1||e2|sin.
        const float scale = std::max(dot(edge1, edge1), dot(edge2, edge2));
        if (dot(faceNormal, faceNormal) <= 1e-8f * scale * scale) continue;

        accumulated[indices[i + 0]] += faceNormal;
        accumulated[indices[i + 1]] += faceNormal;
        accumulated[indices[i + 2]] += faceNormal;
    }

    // A vertex touched only by degenerate faces keeps the normal it came in with
    // rather than being zeroed.
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        if (dot(accumulated[i], accumulated[i]) > 0.0f)
            vertices[i].normal = normalize(accumulated[i]);
    }
}

void Mesh::flipWinding() noexcept {
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) std::swap(indices[i + 1], indices[i + 2]);
    for (Vertex& v : vertices) v.normal = -v.normal;
}

void Mesh::setColor(const Vec3& color) noexcept {
    for (Vertex& v : vertices) v.color = color;
}

void Mesh::transform(const Mat4& matrix) noexcept {
    const Mat4 normalMat = normalMatrix(matrix);
    for (Vertex& v : vertices) {
        v.position = transformPoint(matrix, v.position);
        v.normal = normalize(transformDirection(normalMat, v.normal));
    }
}

void Mesh::normalizeToUnitSize(float targetSize) noexcept {
    if (vertices.empty()) return;

    const Bounds b = bounds();
    const Vec3 e = b.extent();
    const float largest = std::max({e.x, e.y, e.z});
    if (largest <= 1e-8f) return;

    const Vec3 c = b.center();
    const float scale = targetSize / largest;
    for (Vertex& v : vertices) v.position = (v.position - c) * scale;
}

void Mesh::append(const Mesh& other) {
    const std::uint32_t base = static_cast<std::uint32_t>(vertices.size());
    vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
    indices.reserve(indices.size() + other.indices.size());
    for (std::uint32_t index : other.indices) indices.push_back(base + index);
}

Mesh Mesh::cube(float size) {
    const float h = size * 0.5f;

    // Outward normal plus two in-plane axes with cross(u, v) == normal, which
    // makes the (-u,-v) (+u,-v) (+u,+v) (-u,+v) corner order CCW.
    struct Face {
        Vec3 normal, u, v;
    };
    static constexpr Face kFaces[6] = {
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
        {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},
    };
    static constexpr float kCorner[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};

    Mesh mesh;
    mesh.vertices.reserve(24);
    mesh.indices.reserve(36);

    for (const Face& face : kFaces) {
        const std::uint32_t base = static_cast<std::uint32_t>(mesh.vertices.size());
        for (const auto& corner : kCorner) {
            Vertex vert;
            vert.position = face.normal * h + face.u * (corner[0] * h) + face.v * (corner[1] * h);
            vert.normal = face.normal;
            vert.uv = {corner[0] * 0.5f + 0.5f, corner[1] * 0.5f + 0.5f};
            mesh.vertices.push_back(vert);
        }
        mesh.indices.insert(mesh.indices.end(),
                            {base, base + 1, base + 2, base, base + 2, base + 3});
    }
    return mesh;
}

Mesh Mesh::plane(float size, int subdivisions) {
    const int n = std::max(1, subdivisions);
    const float h = size * 0.5f;
    const float step = size / static_cast<float>(n);

    Mesh mesh;
    mesh.vertices.reserve(static_cast<std::size_t>(n + 1) * (n + 1));

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            Vertex v;
            v.position = {-h + static_cast<float>(i) * step, 0.0f,
                          -h + static_cast<float>(j) * step};
            v.normal = {0.0f, 1.0f, 0.0f};
            v.uv = {static_cast<float>(i) / static_cast<float>(n),
                    static_cast<float>(j) / static_cast<float>(n)};
            mesh.vertices.push_back(v);
        }
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) emitQuad(mesh.indices, i, j, n + 1);

    return mesh;
}

Mesh Mesh::uvSphere(float radius, int stacks, int slices) {
    const int st = std::max(2, stacks);
    const int sl = std::max(3, slices);

    Mesh mesh;
    mesh.vertices.reserve(static_cast<std::size_t>(st + 1) * (sl + 1));

    for (int i = 0; i <= st; ++i) {
        const float phi = kPi * static_cast<float>(i) / static_cast<float>(st);
        const float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
        for (int j = 0; j <= sl; ++j) {
            const float theta = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(sl);
            Vertex v;
            v.normal = {sinPhi * std::cos(theta), cosPhi, sinPhi * std::sin(theta)};
            v.position = v.normal * radius;
            v.uv = {static_cast<float>(j) / static_cast<float>(sl),
                    static_cast<float>(i) / static_cast<float>(st)};
            mesh.vertices.push_back(v);
        }
    }
    for (int i = 0; i < st; ++i)
        for (int j = 0; j < sl; ++j) emitQuad(mesh.indices, i, j, sl + 1);

    return mesh;
}

Mesh Mesh::torus(float majorRadius, float minorRadius, int majorSegments, int minorSegments) {
    const int maj = std::max(3, majorSegments);
    const int min_ = std::max(3, minorSegments);

    Mesh mesh;
    mesh.vertices.reserve(static_cast<std::size_t>(maj + 1) * (min_ + 1));

    for (int i = 0; i <= maj; ++i) {
        const float u = 2.0f * kPi * static_cast<float>(i) / static_cast<float>(maj);
        const float cosU = std::cos(u), sinU = std::sin(u);
        for (int j = 0; j <= min_; ++j) {
            const float v = 2.0f * kPi * static_cast<float>(j) / static_cast<float>(min_);
            const float cosV = std::cos(v), sinV = std::sin(v);

            Vertex vert;
            vert.normal = {cosV * cosU, sinV, cosV * sinU};
            vert.position = {(majorRadius + minorRadius * cosV) * cosU, minorRadius * sinV,
                             (majorRadius + minorRadius * cosV) * sinU};
            vert.uv = {static_cast<float>(i) / static_cast<float>(maj),
                       static_cast<float>(j) / static_cast<float>(min_)};
            mesh.vertices.push_back(vert);
        }
    }
    for (int i = 0; i < maj; ++i)
        for (int j = 0; j < min_; ++j) emitQuad(mesh.indices, i, j, min_ + 1);

    return mesh;
}

std::optional<Mesh> Mesh::loadObj(const std::string& path, std::string* error) {
    std::ifstream file(path);
    if (!file) {
        if (error) *error = "could not open '" + path + "'";
        return std::nullopt;
    }

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;

    Mesh mesh;
    // OBJ indexes position/uv/normal independently, so each distinct triple
    // becomes one vertex in the output buffer.
    std::unordered_map<std::string, std::uint32_t> vertexCache;
    bool hasNormals = false;

    auto resolve = [&](int index, std::size_t count) -> int {
        if (index > 0) return index - 1;                       // 1-based
        if (index < 0) return static_cast<int>(count) + index;  // relative to end
        return -1;
    };

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream stream(line);
        std::string tag;
        stream >> tag;

        if (tag == "v") {
            Vec3 p;
            stream >> p.x >> p.y >> p.z;
            positions.push_back(p);
        } else if (tag == "vn") {
            Vec3 n;
            stream >> n.x >> n.y >> n.z;
            normals.push_back(n);
        } else if (tag == "vt") {
            Vec2 t;
            stream >> t.x >> t.y;
            uvs.push_back(t);
        } else if (tag == "f") {
            std::vector<std::uint32_t> face;
            std::string token;
            while (stream >> token) {
                auto cached = vertexCache.find(token);
                if (cached != vertexCache.end()) {
                    face.push_back(cached->second);
                    continue;
                }

                // token is "p", "p/t", "p//n" or "p/t/n".
                int refs[3] = {0, 0, 0};
                std::size_t start = 0;
                for (int slot = 0; slot < 3 && start <= token.size(); ++slot) {
                    const std::size_t slash = token.find('/', start);
                    const std::string part = token.substr(
                        start, slash == std::string::npos ? std::string::npos : slash - start);
                    if (!part.empty()) refs[slot] = std::stoi(part);
                    if (slash == std::string::npos) break;
                    start = slash + 1;
                }

                Vertex vert;
                const int pi = resolve(refs[0], positions.size());
                if (pi < 0 || pi >= static_cast<int>(positions.size())) continue;
                vert.position = positions[static_cast<std::size_t>(pi)];

                const int ti = resolve(refs[1], uvs.size());
                if (ti >= 0 && ti < static_cast<int>(uvs.size()))
                    vert.uv = uvs[static_cast<std::size_t>(ti)];

                const int ni = resolve(refs[2], normals.size());
                if (ni >= 0 && ni < static_cast<int>(normals.size())) {
                    vert.normal = normals[static_cast<std::size_t>(ni)];
                    hasNormals = true;
                }

                const auto newIndex = static_cast<std::uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back(vert);
                vertexCache.emplace(token, newIndex);
                face.push_back(newIndex);
            }

            // Fan-triangulate convex polygons of any size.
            for (std::size_t k = 2; k < face.size(); ++k) {
                mesh.indices.push_back(face[0]);
                mesh.indices.push_back(face[k - 1]);
                mesh.indices.push_back(face[k]);
            }
        }
    }

    if (mesh.vertices.empty() || mesh.indices.empty()) {
        if (error) *error = "'" + path + "' contained no triangles";
        return std::nullopt;
    }
    if (!hasNormals) mesh.recomputeNormals();

    return mesh;
}

}  // namespace sr
