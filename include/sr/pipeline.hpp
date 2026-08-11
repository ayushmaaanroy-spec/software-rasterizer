// The rasterization pipeline itself.
//
// draw() runs three stages:
//   1. geometry  - vertex shading, frustum clipping, back-face culling and the
//                  viewport transform, producing screen-space triangles;
//   2. binning   - each triangle is filed into every 64x64 tile it touches;
//   3. raster    - tiles are claimed by worker threads and scan-converted.
//
// Binning by tile is what makes the raster stage safe to thread: a tile is
// owned by exactly one worker, so no two threads ever touch the same pixel of
// the colour or depth buffer, and the result is bit-identical to the
// single-threaded path regardless of thread count.
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

// Anything the vertex stage wants interpolated across a triangle: it just has
// to be a vector space over float.
template <class T>
concept VaryingType = std::default_initializable<T> && requires(const T& a, const T& b, float s) {
    { a * s } -> std::convertible_to<T>;
    { a + b } -> std::convertible_to<T>;
};

// The shader interface, checked at compile time instead of through inheritance
// and virtual dispatch, so the fragment stage inlines into the inner loop.
template <class S>
concept ShaderProgram = requires {
    typename S::VertexIn;
    typename S::Varyings;
} && VaryingType<typename S::Varyings> &&
    requires(const S& shader, const typename S::VertexIn& in, typename S::Varyings& out,
             const typename S::Varyings& interpolated) {
        // Returns the clip-space position and fills `out` with varyings.
        { shader.vertex(in, out) } -> std::same_as<Vec4>;
        // Returns linear RGB + alpha, or nullopt to discard the fragment.
        { shader.fragment(interpolated) } -> std::same_as<std::optional<Vec4>>;
    };

enum class CullMode { None, Back, Front };
enum class BlendMode { Opaque, Alpha };

struct RenderStats {
    std::uint64_t trianglesSubmitted = 0;
    std::uint64_t trianglesCulled = 0;      // removed by back/front-face culling
    std::uint64_t trianglesClipped = 0;     // straddled the frustum, had to be split
    std::uint64_t trianglesRejected = 0;    // entirely outside the frustum
    std::uint64_t trianglesRasterized = 0;  // reached the raster stage post-clip
    std::uint64_t fragmentsCovered = 0;     // passed the coverage test
    std::uint64_t fragmentsShaded = 0;      // survived the depth test, shader ran
    std::uint64_t fragmentsWritten = 0;     // actually written to the framebuffer

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
    bool colorWrite = true;  // false gives a depth-only pass, e.g. a shadow map
    BlendMode blend = BlendMode::Opaque;
    int threads = 0;  // 0 -> std::thread::hardware_concurrency()
};

namespace detail {

inline constexpr int kTileSize = 64;

// A triangle after the viewport transform. z is window depth in [0, 1]; invW is
// kept per-vertex so the raster stage can undo the perspective divide when
// interpolating varyings.
template <class V>
struct ScreenTriangle {
    Vec3 screen[3];
    float invW[3];
    V varyings[3];
    int minX, minY, maxX, maxY;
};

// Twice the signed area of (a, b, p). Screen space has y growing downward, so a
// positive result means the winding is clockwise on screen.
[[nodiscard]] inline float edgeFunction(const Vec3& a, const Vec3& b, float px, float py) noexcept {
    return (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
}

// Fill rule for shared edges: a pixel centre landing exactly on an edge belongs
// to the triangle only if that edge is a top or left edge. Without this, seams
// are either double-shaded (visible with blending) or dropped entirely.
// Derived for positive-area (clockwise on screen) winding with y downward.
[[nodiscard]] inline bool isTopLeftEdge(const Vec3& a, const Vec3& b) noexcept {
    if (a.y == b.y) return b.x > a.x;  // horizontal edge with the interior below
    return b.y < a.y;                  // edge running up the screen
}

[[nodiscard]] inline int resolveThreadCount(int requested, std::size_t workItems) noexcept {
    int count = requested;
    if (count <= 0) count = static_cast<int>(std::thread::hardware_concurrency());
    if (count <= 0) count = 1;
    if (static_cast<std::size_t>(count) > workItems) count = static_cast<int>(workItems);
    return count < 1 ? 1 : count;
}

// Scan-converts one triangle, restricted to the pixel rectangle
// [clipX0, clipX1] x [clipY0, clipY1] (the owning tile, or the whole target).
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

    // Guaranteed positive by the geometry stage, which swaps vertices as needed.
    const float area = edgeFunction(p0, p1, p2.x, p2.y);
    const float invArea = 1.0f / area;

    const bool topLeft0 = isTopLeftEdge(p1, p2);
    const bool topLeft1 = isTopLeftEdge(p2, p0);
    const bool topLeft2 = isTopLeftEdge(p0, p1);

    // The edge function is affine in x, so stepping across a scanline is a add.
    // Each row restarts from an exact evaluation to stop error accumulating.
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

            // Window-space z is affine in screen space, so it interpolates
            // directly with the screen-space barycentrics.
            const float depth = b0 * p0.z + b1 * p1.z + b2 * p2.z;
            float& dstDepth = fb.depthAt(x, y);
            if (config.depthTest && depth > dstDepth) continue;
            ++stats.fragmentsShaded;

            // Everything else must be perspective-corrected: interpolate 1/w
            // linearly, then weight each vertex by its own 1/w.
            const float interpolatedInvW = b0 * tri.invW[0] + b1 * tri.invW[1] + b2 * tri.invW[2];
            const float renormalize = 1.0f / interpolatedInvW;
            const float c0 = b0 * tri.invW[0] * renormalize;
            const float c1 = b1 * tri.invW[1] * renormalize;
            const float c2 = b2 * tri.invW[2] * renormalize;

            const typename S::Varyings varyings =
                tri.varyings[0] * c0 + tri.varyings[1] * c1 + tri.varyings[2] * c2;

            const std::optional<Vec4> shaded = shader.fragment(varyings);
            if (!shaded) continue;  // discarded by the shader

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
    // Dispatches fn(0..count-1) across the pool, creating it on first use so a
    // purely single-threaded caller never spawns a thread.
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

    // ------------------------------------------------------------ geometry
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

            // Fan-triangulate the clipped polygon around its first vertex.
            for (int k = 2; k < count; ++k) {
                const ClipVertex<V>* corners[3] = {&polygon[0],
                                                   &polygon[static_cast<std::size_t>(k - 1)],
                                                   &polygon[static_cast<std::size_t>(k)]};
                Triangle tri;
                for (int c = 0; c < 3; ++c) {
                    const Vec4& clip = corners[c]->position;
                    const float invW = 1.0f / clip.w;
                    // Perspective divide, then NDC -> pixels. y is flipped
                    // because row 0 of the framebuffer is the top of the image.
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

                // Front faces are counter-clockwise in NDC, which the y flip
                // turns into a negative screen-space area.
                const bool frontFacing = area < 0.0f;
                if ((config.cull == CullMode::Back && !frontFacing) ||
                    (config.cull == CullMode::Front && frontFacing)) {
                    ++stats.trianglesCulled;
                    continue;
                }

                // The raster stage assumes positive area; swapping two corners
                // flips the winding without changing the covered pixels.
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

    // Concatenating in thread order keeps submission order, so overlapping
    // coplanar geometry resolves the same way every run.
    std::size_t totalTriangles = 0;
    for (const std::vector<Triangle>& chunk : perThreadTriangles) totalTriangles += chunk.size();

    std::vector<Triangle> triangles;
    triangles.reserve(totalTriangles);
    for (const std::vector<Triangle>& chunk : perThreadTriangles)
        triangles.insert(triangles.end(), chunk.begin(), chunk.end());

    // -------------------------------------------------------------- binning
    // Tiles are a fixed decomposition of the target, independent of how many
    // workers there are. Because a triangle is always scan-converted starting
    // from its tile's left edge, every fragment sees the same arithmetic at any
    // thread count -- so the rendered image is bit-identical, not merely
    // equivalent. Serial rendering is just this loop with one worker.
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

        // --------------------------------------------------------- raster
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

                // Bin order is global submission order, so overlapping coplanar
                // surfaces resolve identically however the tiles are scheduled.
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
