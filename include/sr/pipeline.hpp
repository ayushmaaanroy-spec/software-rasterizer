// draw() runs three stages: geometry (vertex shading, clipping, culling,
// viewport transform), binning (file each triangle into the tiles it touches),
// and raster (workers claim tiles and scan-convert them).
#pragma once

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "sr/clip.hpp"
#include "sr/framebuffer.hpp"
#include "sr/math.hpp"
#include "sr/thread_pool.hpp"

namespace sr {

// Varyings only need to be a vector space over float.
template <class T>
concept VaryingType = std::default_initializable<T> && requires(const T& a, const T& b, float s) {
    { a * s } -> std::convertible_to<T>;
    { a + b } -> std::convertible_to<T>;
};

// Checked at compile time rather than through a vtable, so fragment() inlines
// into the raster loop.
template <class S>
concept ShaderProgram = requires {
    typename S::VertexIn;
    typename S::Varyings;
} && VaryingType<typename S::Varyings> &&
    requires(const S& shader, const typename S::VertexIn& in, typename S::Varyings& out,
             const typename S::Varyings& interpolated) {
        { shader.vertex(in, out) } -> std::same_as<Vec4>;
        // Linear RGB + alpha, or nullopt to discard.
        { shader.fragment(interpolated) } -> std::same_as<std::optional<Vec4>>;
    };

enum class CullMode { None, Back, Front };
enum class BlendMode { Opaque, Alpha };

struct RenderStats {
    std::uint64_t trianglesSubmitted = 0;
    std::uint64_t trianglesCulled = 0;
    std::uint64_t trianglesClipped = 0;     // straddled the frustum, had to be split
    std::uint64_t trianglesRejected = 0;    // entirely outside the frustum
    std::uint64_t trianglesRasterized = 0;  // post-clip, so can exceed the input count
    std::uint64_t fragmentsCovered = 0;
    std::uint64_t fragmentsShaded = 0;  // passed the depth test, shader ran
    std::uint64_t fragmentsWritten = 0;

    void add(const RenderStats& o) noexcept {
        trianglesSubmitted += o.trianglesSubmitted;
        trianglesCulled += o.trianglesCulled;
        trianglesClipped += o.trianglesClipped;
        trianglesRejected += o.trianglesRejected;
        trianglesRasterized += o.trianglesRasterized;
        fragmentsCovered += o.fragmentsCovered;
        fragmentsShaded += o.fragmentsShaded;
        fragmentsWritten += o.fragmentsWritten;
    }
};

struct PipelineConfig {
    CullMode cull = CullMode::Back;
    bool depthTest = true;
    bool depthWrite = true;
    bool colorWrite = true;  // false for a depth-only pass such as a shadow map
    BlendMode blend = BlendMode::Opaque;
    int threads = 0;  // 0 means hardware_concurrency()
};

namespace detail {

inline constexpr int kTileSize = 64;

template <class V>
struct ScreenTriangle {
    Vec3 screen[3];  // x,y in pixels, z is window depth in [0,1]
    float invW[3];   // kept so the raster stage can undo the perspective divide
    V varyings[3];
    int minX, minY, maxX, maxY;
};

// Twice the signed area of (a, b, p). y grows downward, so positive means
// clockwise on screen.
[[nodiscard]] inline float edgeFunction(const Vec3& a, const Vec3& b, float px, float py) noexcept {
    return (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
}

// Top-left fill rule: a pixel center exactly on a shared edge goes to one
// triangle only, never both or neither. Assumes positive area and y downward.
[[nodiscard]] inline bool isTopLeftEdge(const Vec3& a, const Vec3& b) noexcept {
    if (a.y == b.y) return b.x > a.x;
    return b.y < a.y;
}

[[nodiscard]] inline int resolveThreadCount(int requested, std::size_t workItems) noexcept {
    int count = requested;
    if (count <= 0) count = static_cast<int>(std::thread::hardware_concurrency());
    if (count <= 0) count = 1;
    if (static_cast<std::size_t>(count) > workItems) count = static_cast<int>(workItems);
    return count < 1 ? 1 : count;
}

// Scan-converts one triangle, clipped to the given pixel rectangle (its tile).
template <ShaderProgram S>
void rasterizeTriangle(const S& shader, Framebuffer& fb, const PipelineConfig& config,
                       const ScreenTriangle<typename S::Varyings>& tri, int clipX0, int clipY0,
                       int clipX1, int clipY1, RenderStats& stats) {
    const int x0 = std::max(tri.minX, clipX0);
    const int x1 = std::min(tri.maxX, clipX1);
    const int y0 = std::max(tri.minY, clipY0);
    const int y1 = std::min(tri.maxY, clipY1);
    if (x0 > x1 || y0 > y1) return;

    const Vec3& p0 = tri.screen[0];
    const Vec3& p1 = tri.screen[1];
    const Vec3& p2 = tri.screen[2];

    const float area = edgeFunction(p0, p1, p2.x, p2.y);  // positive, geometry stage ensures it
    const float invArea = 1.0f / area;

    const bool topLeft0 = isTopLeftEdge(p1, p2);
    const bool topLeft1 = isTopLeftEdge(p2, p0);
    const bool topLeft2 = isTopLeftEdge(p0, p1);

    // Edge functions are affine in x, so a scanline steps by a constant. Rows
    // restart from an exact evaluation so error cannot accumulate.
    const float stepX0 = -(p2.y - p1.y);
    const float stepX1 = -(p0.y - p2.y);
    const float stepX2 = -(p1.y - p0.y);

    for (int y = y0; y <= y1; ++y) {
        const float py = static_cast<float>(y) + 0.5f;
        const float px = static_cast<float>(x0) + 0.5f;
        float w0 = edgeFunction(p1, p2, px, py);
        float w1 = edgeFunction(p2, p0, px, py);
        float w2 = edgeFunction(p0, p1, px, py);

        for (int x = x0; x <= x1; ++x, w0 += stepX0, w1 += stepX1, w2 += stepX2) {
            if (!(w0 > 0.0f || (w0 == 0.0f && topLeft0))) continue;
            if (!(w1 > 0.0f || (w1 == 0.0f && topLeft1))) continue;
            if (!(w2 > 0.0f || (w2 == 0.0f && topLeft2))) continue;
            ++stats.fragmentsCovered;

            const float b0 = w0 * invArea;
            const float b1 = w1 * invArea;
            const float b2 = w2 * invArea;

            // Window z is affine in screen space, so plain barycentrics are exact.
            const float depth = b0 * p0.z + b1 * p1.z + b2 * p2.z;
            float& dstDepth = fb.depthAt(x, y);
            if (config.depthTest && depth > dstDepth) continue;
            ++stats.fragmentsShaded;

            // Varyings are not. Interpolate 1/w, then reweight by each vertex's own.
            const float interpolatedInvW = b0 * tri.invW[0] + b1 * tri.invW[1] + b2 * tri.invW[2];
            const float renormalize = 1.0f / interpolatedInvW;
            const float c0 = b0 * tri.invW[0] * renormalize;
            const float c1 = b1 * tri.invW[1] * renormalize;
            const float c2 = b2 * tri.invW[2] * renormalize;

            const typename S::Varyings varyings =
                tri.varyings[0] * c0 + tri.varyings[1] * c1 + tri.varyings[2] * c2;

            const std::optional<Vec4> shaded = shader.fragment(varyings);
            if (!shaded) continue;

            if (config.colorWrite) {
                Vec3 color = shaded->xyz();
                if (config.blend == BlendMode::Alpha) {
                    const float alpha = clampf(shaded->w, 0.0f, 1.0f);
                    color = lerp(fb.colorAt(x, y), color, alpha);
                }
                fb.colorAt(x, y) = color;
            }
            if (config.depthWrite) dstDepth = depth;
            ++stats.fragmentsWritten;
        }
    }
}

}  // namespace detail

class Rasterizer {
public:
    explicit Rasterizer(Framebuffer& target) noexcept : target_(&target) {}

    [[nodiscard]] Framebuffer& target() noexcept { return *target_; }
    [[nodiscard]] const RenderStats& stats() const noexcept { return stats_; }
    void resetStats() noexcept { stats_ = {}; }

    template <ShaderProgram S>
    void draw(const S& shader, std::span<const typename S::VertexIn> vertices,
              std::span<const std::uint32_t> indices, const PipelineConfig& config = {});

private:
    // Pool is built on first use, so single-threaded callers never spawn a thread.
    void dispatch(int count, const std::function<void(int)>& fn) {
        if (count <= 1) {
            fn(0);
            return;
        }
        if (!pool_ || pool_->width() < count) pool_ = std::make_unique<ThreadPool>(count - 1);
        pool_->run(count, fn);
    }

    Framebuffer* target_;
    RenderStats stats_;
    std::unique_ptr<ThreadPool> pool_;
};

template <ShaderProgram S>
void Rasterizer::draw(const S& shader, std::span<const typename S::VertexIn> vertices,
                      std::span<const std::uint32_t> indices, const PipelineConfig& config) {
    using V = typename S::Varyings;
    using Triangle = detail::ScreenTriangle<V>;

    const std::size_t inputTriangles = indices.size() / 3;
    if (inputTriangles == 0) return;

    Framebuffer& fb = *target_;
    const int width = fb.width();
    const int height = fb.height();
    const float fWidth = static_cast<float>(width);
    const float fHeight = static_cast<float>(height);

    const int threadCount = detail::resolveThreadCount(config.threads, inputTriangles);
    std::vector<std::vector<Triangle>> perThreadTriangles(static_cast<std::size_t>(threadCount));
    std::vector<RenderStats> perThreadStats(static_cast<std::size_t>(threadCount));

    auto geometryStage = [&](int t) {
        const std::size_t begin = inputTriangles * static_cast<std::size_t>(t) / threadCount;
        const std::size_t end = inputTriangles * static_cast<std::size_t>(t + 1) / threadCount;

        std::vector<Triangle>& out = perThreadTriangles[static_cast<std::size_t>(t)];
        RenderStats& stats = perThreadStats[static_cast<std::size_t>(t)];
        out.reserve(end - begin);

        ClipPolygon<V> polygon;
        ClipPolygon<V> scratch;

        for (std::size_t i = begin; i < end; ++i) {
            ++stats.trianglesSubmitted;

            for (int k = 0; k < 3; ++k) {
                const auto& vertexIn = vertices[indices[i * 3 + static_cast<std::size_t>(k)]];
                polygon[static_cast<std::size_t>(k)].position =
                    shader.vertex(vertexIn, polygon[static_cast<std::size_t>(k)].varyings);
            }

            int count = 3;
            if (!triviallyAccepted(polygon[0], polygon[1], polygon[2])) {
                if (triviallyRejected(polygon[0], polygon[1], polygon[2])) {
                    ++stats.trianglesRejected;
                    continue;
                }
                count = clipPolygon(polygon, 3, scratch);
                if (count == 0) {
                    ++stats.trianglesRejected;
                    continue;
                }
                ++stats.trianglesClipped;
            }

            for (int k = 2; k < count; ++k) {  // fan-triangulate the clipped polygon
                const ClipVertex<V>* corners[3] = {&polygon[0],
                                                   &polygon[static_cast<std::size_t>(k - 1)],
                                                   &polygon[static_cast<std::size_t>(k)]};
                Triangle tri;
                for (int c = 0; c < 3; ++c) {
                    const Vec4& clip = corners[c]->position;
                    const float invW = 1.0f / clip.w;
                    // Perspective divide, then NDC to pixels. y flips because row
                    // 0 is the top of the image.
                    tri.screen[c] = {(clip.x * invW * 0.5f + 0.5f) * fWidth,
                                     (0.5f - clip.y * invW * 0.5f) * fHeight,
                                     clip.z * invW * 0.5f + 0.5f};
                    tri.invW[c] = invW;
                    tri.varyings[c] = corners[c]->varyings;
                }

                const float area =
                    detail::edgeFunction(tri.screen[0], tri.screen[1], tri.screen[2].x,
                                         tri.screen[2].y);
                if (!(std::fabs(area) > 1e-9f)) continue;  // degenerate or NaN

                // Front faces are CCW in NDC, which the y flip makes negative here.
                const bool frontFacing = area < 0.0f;
                if ((config.cull == CullMode::Back && !frontFacing) ||
                    (config.cull == CullMode::Front && frontFacing)) {
                    ++stats.trianglesCulled;
                    continue;
                }

                // Raster wants positive area. Swapping two corners flips the
                // winding without changing coverage.
                if (area < 0.0f) {
                    std::swap(tri.screen[1], tri.screen[2]);
                    std::swap(tri.invW[1], tri.invW[2]);
                    std::swap(tri.varyings[1], tri.varyings[2]);
                }

                float minX = tri.screen[0].x, maxX = tri.screen[0].x;
                float minY = tri.screen[0].y, maxY = tri.screen[0].y;
                for (int c = 1; c < 3; ++c) {
                    minX = std::min(minX, tri.screen[c].x);
                    maxX = std::max(maxX, tri.screen[c].x);
                    minY = std::min(minY, tri.screen[c].y);
                    maxY = std::max(maxY, tri.screen[c].y);
                }
                tri.minX = std::max(0, static_cast<int>(std::floor(minX)));
                tri.minY = std::max(0, static_cast<int>(std::floor(minY)));
                tri.maxX = std::min(width - 1, static_cast<int>(std::ceil(maxX)));
                tri.maxY = std::min(height - 1, static_cast<int>(std::ceil(maxY)));
                if (tri.minX > tri.maxX || tri.minY > tri.maxY) continue;

                ++stats.trianglesRasterized;
                out.push_back(tri);
            }
        }
    };
    dispatch(threadCount, geometryStage);

    // Concatenating in thread order preserves submission order.
    std::size_t totalTriangles = 0;
    for (const std::vector<Triangle>& chunk : perThreadTriangles) totalTriangles += chunk.size();

    std::vector<Triangle> triangles;
    triangles.reserve(totalTriangles);
    for (const std::vector<Triangle>& chunk : perThreadTriangles)
        triangles.insert(triangles.end(), chunk.begin(), chunk.end());

    // One worker owns a tile, so no two threads touch the same pixel and there
    // are no atomics on the buffers. The tile grid does not depend on the thread
    // count and triangles always start scanning from their tile's left edge, so
    // the output is bit-identical at any thread count.
    if (totalTriangles != 0) {
        const int tilesX = (width + detail::kTileSize - 1) / detail::kTileSize;
        const int tilesY = (height + detail::kTileSize - 1) / detail::kTileSize;
        const int tileCount = tilesX * tilesY;

        std::vector<std::vector<std::uint32_t>> bins(static_cast<std::size_t>(tileCount));
        for (std::uint32_t i = 0; i < triangles.size(); ++i) {
            const Triangle& tri = triangles[i];
            for (int ty = tri.minY / detail::kTileSize; ty <= tri.maxY / detail::kTileSize; ++ty)
                for (int tx = tri.minX / detail::kTileSize; tx <= tri.maxX / detail::kTileSize; ++tx)
                    bins[static_cast<std::size_t>(ty) * static_cast<std::size_t>(tilesX) +
                         static_cast<std::size_t>(tx)]
                        .push_back(i);
        }

        std::atomic<int> nextTile{0};
        auto rasterStage = [&](int t) {
            RenderStats& stats = perThreadStats[static_cast<std::size_t>(t)];
            for (int tile = nextTile.fetch_add(1, std::memory_order_relaxed); tile < tileCount;
                 tile = nextTile.fetch_add(1, std::memory_order_relaxed)) {
                const std::vector<std::uint32_t>& bin = bins[static_cast<std::size_t>(tile)];
                if (bin.empty()) continue;

                const int tileX = (tile % tilesX) * detail::kTileSize;
                const int tileY = (tile / tilesX) * detail::kTileSize;
                const int tileX1 = std::min(tileX + detail::kTileSize - 1, width - 1);
                const int tileY1 = std::min(tileY + detail::kTileSize - 1, height - 1);

                for (std::uint32_t index : bin) {
                    detail::rasterizeTriangle(shader, fb, config, triangles[index], tileX, tileY,
                                              tileX1, tileY1, stats);
                }
            }
        };
        dispatch(std::min(threadCount, tileCount), rasterStage);
    }

    for (const RenderStats& s : perThreadStats) stats_.add(s);
}

}  // namespace sr
