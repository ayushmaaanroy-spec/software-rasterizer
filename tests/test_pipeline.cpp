#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "check.hpp"
#include "sr/camera.hpp"
#include "sr/clip.hpp"
#include "sr/framebuffer.hpp"
#include "sr/mesh.hpp"
#include "sr/pipeline.hpp"
#include "sr/shaders.hpp"

using namespace sr;

namespace {

// Passes the vertex colour straight through, so a rendered pixel reports the
// interpolated varying exactly.
struct ProbeShader {
    using VertexIn = Vertex;

    struct Varyings {
        Vec3 value;

        [[nodiscard]] Varyings operator*(float s) const noexcept { return {value * s}; }
        [[nodiscard]] Varyings operator+(const Varyings& o) const noexcept {
            return {value + o.value};
        }
    };

    Mat4 viewProjection = Mat4::identity();
    // When set, fragments whose interpolated red channel is below this are discarded.
    float discardBelow = -1.0f;

    [[nodiscard]] Vec4 vertex(const VertexIn& in, Varyings& out) const noexcept {
        out.value = in.color;
        return viewProjection * Vec4(in.position, 1.0f);
    }

    [[nodiscard]] std::optional<Vec4> fragment(const Varyings& in) const noexcept {
        if (in.value.x < discardBelow) return std::nullopt;
        return Vec4(in.value, 1.0f);
    }
};

std::vector<Vertex> quad(float z, const Vec3& color, float halfSize = 1.0f) {
    std::vector<Vertex> v(4);
    v[0].position = {-halfSize, -halfSize, z};
    v[1].position = {halfSize, -halfSize, z};
    v[2].position = {halfSize, halfSize, z};
    v[3].position = {-halfSize, halfSize, z};
    for (Vertex& vertex : v) vertex.color = color;
    return v;
}

const std::vector<std::uint32_t> kQuadIndices = {0, 1, 2, 0, 2, 3};

}  // namespace

// A screen-filling quad must cover every pixel exactly once. Any double
// coverage along the shared diagonal, or any dropped pixel, shows up here --
// this is the assertion that pins down the top-left fill rule.
TEST(fullscreen_quad_covers_every_pixel_exactly_once) {
    constexpr int kWidth = 64, kHeight = 48;
    Framebuffer fb(kWidth, kHeight);
    fb.clear(Vec3{0.0f}, 1.0f);

    Rasterizer raster(fb);
    PipelineConfig config;
    config.cull = CullMode::None;
    config.threads = 1;

    const std::vector<Vertex> verts = quad(0.0f, Vec3{1.0f});
    raster.draw(ProbeShader{}, verts, kQuadIndices, config);

    CHECK(raster.stats().trianglesRasterized == 2);
    CHECK(raster.stats().trianglesClipped == 0);
    CHECK(raster.stats().fragmentsCovered == static_cast<std::uint64_t>(kWidth) * kHeight);
    CHECK(raster.stats().fragmentsWritten == static_cast<std::uint64_t>(kWidth) * kHeight);

    bool allWritten = true;
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth; ++x)
            if (fb.colorAt(x, y).x != 1.0f) allWritten = false;
    CHECK(allWritten);
}

TEST(back_face_culling_keys_off_winding) {
    Framebuffer fb(64, 64);
    std::vector<Vertex> verts(3);
    verts[0].position = {-0.5f, -0.5f, 0.0f};
    verts[1].position = {0.5f, -0.5f, 0.0f};
    verts[2].position = {0.0f, 0.5f, 0.0f};
    for (Vertex& v : verts) v.color = Vec3{1.0f};

    const std::vector<std::uint32_t> frontFacing = {0, 1, 2};  // counter-clockwise in NDC
    const std::vector<std::uint32_t> backFacing = {0, 2, 1};

    PipelineConfig config;
    config.cull = CullMode::Back;

    {
        Rasterizer raster(fb);
        raster.draw(ProbeShader{}, verts, frontFacing, config);
        CHECK(raster.stats().trianglesCulled == 0);
        CHECK(raster.stats().trianglesRasterized == 1);
        CHECK(raster.stats().fragmentsCovered > 0);
    }
    {
        Rasterizer raster(fb);
        raster.draw(ProbeShader{}, verts, backFacing, config);
        CHECK(raster.stats().trianglesCulled == 1);
        CHECK(raster.stats().trianglesRasterized == 0);
        CHECK(raster.stats().fragmentsCovered == 0);
    }
    {
        // Reversed winding still rasterizes when culling is off, and the raster
        // stage must handle the negative signed area by reordering internally.
        Rasterizer raster(fb);
        PipelineConfig noCull = config;
        noCull.cull = CullMode::None;
        raster.draw(ProbeShader{}, verts, backFacing, noCull);
        CHECK(raster.stats().trianglesRasterized == 1);
        CHECK(raster.stats().fragmentsCovered > 0);
    }
    {
        Rasterizer raster(fb);
        PipelineConfig frontCull = config;
        frontCull.cull = CullMode::Front;
        raster.draw(ProbeShader{}, verts, frontFacing, frontCull);
        CHECK(raster.stats().trianglesCulled == 1);
    }
}

TEST(depth_buffer_resolves_occlusion) {
    constexpr int kWidth = 32, kHeight = 32;
    Framebuffer fb(kWidth, kHeight);
    fb.clear(Vec3{0.0f}, 1.0f);

    Rasterizer raster(fb);
    PipelineConfig config;
    config.cull = CullMode::None;
    config.threads = 1;

    // Far surface first, then a nearer one, then the far one again. The nearest
    // surface must win regardless of submission order.
    const std::vector<Vertex> far = quad(0.6f, Vec3{1.0f, 0.0f, 0.0f});
    const std::vector<Vertex> near = quad(-0.6f, Vec3{0.0f, 1.0f, 0.0f}, 0.5f);

    raster.draw(ProbeShader{}, far, kQuadIndices, config);
    raster.draw(ProbeShader{}, near, kQuadIndices, config);
    raster.draw(ProbeShader{}, far, kQuadIndices, config);

    // Centre pixel is covered by the near quad; a corner only by the far one.
    const Vec3 center = fb.colorAt(kWidth / 2, kHeight / 2);
    CHECK_NEAR(center.y, 1.0, 1e-6);
    CHECK_NEAR(center.x, 0.0, 1e-6);
    CHECK_NEAR(fb.depthAt(kWidth / 2, kHeight / 2), 0.2, 1e-5);  // ndc -0.6 -> window 0.2

    const Vec3 corner = fb.colorAt(1, 1);
    CHECK_NEAR(corner.x, 1.0, 1e-6);
    CHECK_NEAR(fb.depthAt(1, 1), 0.8, 1e-5);
}

TEST(depth_test_and_write_can_be_disabled) {
    Framebuffer fb(16, 16);
    fb.clear(Vec3{0.0f}, 1.0f);

    Rasterizer raster(fb);
    PipelineConfig config;
    config.cull = CullMode::None;
    config.depthTest = false;
    config.depthWrite = false;

    raster.draw(ProbeShader{}, quad(-0.6f, Vec3{0.0f, 1.0f, 0.0f}), kQuadIndices, config);
    raster.draw(ProbeShader{}, quad(0.6f, Vec3{1.0f, 0.0f, 0.0f}), kQuadIndices, config);

    // With the depth test off, the last draw wins even though it is further away.
    CHECK_NEAR(fb.colorAt(8, 8).x, 1.0, 1e-6);
    // ...and with depth writes off the buffer is untouched.
    CHECK_NEAR(fb.depthAt(8, 8), 1.0, 1e-6);
}

// Varyings must be interpolated with the perspective divide undone. A ground
// plane receding from the camera makes the difference obvious: affine
// interpolation is off by a factor of three at the sample point.
TEST(varyings_are_perspective_correct) {
    constexpr int kSize = 240;
    Framebuffer fb(kSize, kSize);
    fb.clear(Vec3{0.0f}, 1.0f);

    Rasterizer raster(fb);
    PipelineConfig config;
    config.cull = CullMode::None;
    config.threads = 1;

    // Camera at the origin looking down -Z, so world space is eye space.
    ProbeShader shader;
    shader.viewProjection = perspective(radians(90.0f), 1.0f, 0.25f, 50.0f) *
                            lookAt(Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, -1.0f},
                                   Vec3{0.0f, 1.0f, 0.0f});

    // Floor at y = -1, running from z = -1 (probe 0) to z = -5 (probe 1).
    std::vector<Vertex> verts(4);
    verts[0].position = {-0.9f, -1.0f, -1.0f};
    verts[1].position = {0.9f, -1.0f, -1.0f};
    verts[2].position = {0.9f, -1.0f, -5.0f};
    verts[3].position = {-0.9f, -1.0f, -5.0f};
    verts[0].color = Vec3{0.0f};
    verts[1].color = Vec3{0.0f};
    verts[2].color = Vec3{1.0f};
    verts[3].color = Vec3{1.0f};

    raster.draw(shader, verts, kQuadIndices, config);

    // Near edge sits at ndc.y = -1 (w = 1), far edge at ndc.y = -0.2 (w = 5).
    const int row = static_cast<int>((0.5f + 0.6f * 0.5f) * kSize);
    const float ndcY = 1.0f - 2.0f * (static_cast<float>(row) + 0.5f) / static_cast<float>(kSize);
    const float t = (ndcY + 1.0f) / 0.8f;  // fraction across the quad in screen space
    const float invWNear = 1.0f, invWFar = 0.2f;
    const float expected = (t * invWFar) / (t * invWFar + (1.0f - t) * invWNear);

    const float actual = fb.colorAt(kSize / 2, row).x;
    CHECK_NEAR(actual, expected, 2e-3);
    // Affine interpolation would have produced ~t; confirm we are far from it.
    CHECK(std::fabs(actual - t) > 0.2f);
}

TEST(clipper_handles_the_near_plane) {
    using CV = ClipVertex<ProbeShader::Varyings>;

    const Mat4 proj = perspective(radians(90.0f), 1.0f, 0.5f, 50.0f);
    auto toClip = [&](const Vec3& eyePos) {
        CV v;
        v.position = proj * Vec4(eyePos, 1.0f);
        return v;
    };

    // Entirely in front of the camera and inside the frustum.
    {
        ClipPolygon<ProbeShader::Varyings> poly{}, scratch{};
        poly[0] = toClip({-0.5f, -0.5f, -2.0f});
        poly[1] = toClip({0.5f, -0.5f, -2.0f});
        poly[2] = toClip({0.0f, 0.5f, -2.0f});
        CHECK(triviallyAccepted(poly[0], poly[1], poly[2]));
        CHECK(!triviallyRejected(poly[0], poly[1], poly[2]));
        CHECK(clipPolygon(poly, 3, scratch) == 3);
    }

    // Entirely behind the camera: rejected outright.
    {
        ClipPolygon<ProbeShader::Varyings> poly{}, scratch{};
        poly[0] = toClip({-0.5f, -0.5f, 2.0f});
        poly[1] = toClip({0.5f, -0.5f, 2.0f});
        poly[2] = toClip({0.0f, 0.5f, 3.0f});
        CHECK(!triviallyAccepted(poly[0], poly[1], poly[2]));
        CHECK(triviallyRejected(poly[0], poly[1], poly[2]));
        CHECK(clipPolygon(poly, 3, scratch) == 0);
    }

    // Straddling the eye plane: the clipper has to add vertices, and every one
    // it produces must satisfy every plane.
    {
        ClipPolygon<ProbeShader::Varyings> poly{}, scratch{};
        poly[0] = toClip({-0.4f, -0.4f, -2.0f});
        poly[0].varyings.value = Vec3{0.0f};
        poly[1] = toClip({0.4f, -0.4f, -2.0f});
        poly[1].varyings.value = Vec3{1.0f};
        poly[2] = toClip({0.0f, 0.4f, 1.0f});  // behind the eye
        poly[2].varyings.value = Vec3{1.0f};

        CHECK(!triviallyAccepted(poly[0], poly[1], poly[2]));
        CHECK(!triviallyRejected(poly[0], poly[1], poly[2]));

        const int count = clipPolygon(poly, 3, scratch);
        CHECK(count >= 3);
        CHECK(count <= kMaxClippedVertices);

        bool allInside = true;
        bool varyingsInRange = true;
        for (int i = 0; i < count; ++i) {
            for (ClipPlane plane : kClipPlanes)
                if (planeDistance(plane, poly[static_cast<std::size_t>(i)].position) < -1e-4f)
                    allInside = false;
            const float value = poly[static_cast<std::size_t>(i)].varyings.value.x;
            if (value < -1e-4f || value > 1.0f + 1e-4f) varyingsInRange = false;
        }
        CHECK(allInside);
        CHECK(varyingsInRange);
    }
}

// End-to-end: geometry that crosses the near plane and spills off every side
// must still land inside the framebuffer, with no wrap-around from dividing by
// a negative w.
TEST(clipped_geometry_stays_inside_the_viewport) {
    constexpr int kWidth = 160, kHeight = 120;
    Framebuffer fb(kWidth, kHeight);
    fb.clear(Vec3{0.0f}, 1.0f);

    Rasterizer raster(fb);
    PipelineConfig config;
    config.cull = CullMode::None;
    config.threads = 1;

    ProbeShader shader;
    shader.viewProjection = perspective(radians(70.0f), fb.aspect(), 0.5f, 20.0f) *
                            lookAt(Vec3{0.0f, 0.0f, 0.0f}, Vec3{0.0f, 0.0f, -1.0f},
                                   Vec3{0.0f, 1.0f, 0.0f});

    // A huge slab running from behind the camera out past the far plane.
    std::vector<Vertex> verts(4);
    verts[0].position = {-40.0f, -8.0f, 4.0f};   // behind the eye
    verts[1].position = {40.0f, -8.0f, 4.0f};    // behind the eye
    verts[2].position = {40.0f, 8.0f, -60.0f};   // past the far plane
    verts[3].position = {-40.0f, 8.0f, -60.0f};  // past the far plane
    for (Vertex& v : verts) v.color = Vec3{1.0f};

    raster.draw(shader, verts, kQuadIndices, config);

    CHECK(raster.stats().trianglesClipped == 2);
    CHECK(raster.stats().trianglesRasterized >= 2);
    CHECK(raster.stats().fragmentsCovered > 0);
    // Coverage can never exceed the target, and the depth values written must
    // all be inside the window-space range.
    CHECK(raster.stats().fragmentsCovered <= static_cast<std::uint64_t>(kWidth) * kHeight);

    bool depthInRange = true;
    for (int y = 0; y < kHeight; ++y)
        for (int x = 0; x < kWidth; ++x) {
            const float d = fb.depthAt(x, y);
            if (d < 0.0f || d > 1.0f) depthInRange = false;
        }
    CHECK(depthInRange);
}

TEST(fragment_discard_skips_the_write) {
    Framebuffer fb(32, 32);
    fb.clear(Vec3{0.0f}, 1.0f);

    Rasterizer raster(fb);
    PipelineConfig config;
    config.cull = CullMode::None;
    config.threads = 1;

    // Gradient from 0 on the left to 1 on the right; discard the left half.
    std::vector<Vertex> verts = quad(0.0f, Vec3{0.0f});
    verts[0].color = Vec3{0.0f};
    verts[1].color = Vec3{1.0f};
    verts[2].color = Vec3{1.0f};
    verts[3].color = Vec3{0.0f};

    ProbeShader shader;
    shader.discardBelow = 0.5f;
    raster.draw(shader, verts, kQuadIndices, config);

    const RenderStats& s = raster.stats();
    CHECK(s.fragmentsShaded == static_cast<std::uint64_t>(32) * 32);
    CHECK(s.fragmentsWritten < s.fragmentsShaded);
    CHECK(s.fragmentsWritten > 0);
    // Depth must not be written for discarded fragments.
    CHECK_NEAR(fb.depthAt(2, 16), 1.0, 1e-6);
    CHECK_NEAR(fb.depthAt(29, 16), 0.5, 1e-5);
}

// Tile ownership means threading changes nothing about the output. If this ever
// fails, two workers are touching the same pixel.
TEST(threading_produces_identical_output) {
    const Mesh mesh = Mesh::uvSphere(1.0f, 24, 48);

    Camera camera = Camera::orbit(Vec3{0.0f}, 3.2f, radians(35.0f), radians(20.0f));
    camera.aspect = 4.0f / 3.0f;

    LitShader shader;
    shader.viewProjection = camera.viewProjection();
    shader.eyePosition = camera.eye;
    shader.setModel(rotationY(radians(15.0f)));

    auto render = [&](int threads) {
        Framebuffer fb(200, 150);
        fb.clear(Vec3{0.02f}, 1.0f);
        Rasterizer raster(fb);
        PipelineConfig config;
        config.threads = threads;
        raster.draw(shader, mesh.vertices, mesh.indices, config);
        return fb;
    };

    const Framebuffer single = render(1);
    const Framebuffer multi = render(8);

    bool identical = true;
    for (int y = 0; y < single.height(); ++y)
        for (int x = 0; x < single.width(); ++x) {
            const Vec3& a = single.colorAt(x, y);
            const Vec3& b = multi.colorAt(x, y);
            if (a.x != b.x || a.y != b.y || a.z != b.z) identical = false;
            if (single.depthAt(x, y) != multi.depthAt(x, y)) identical = false;
        }
    CHECK(identical);
}

TEST(framebuffer_downsample_averages_blocks) {
    Framebuffer fb(4, 4);
    fb.clearColor(Vec3{0.0f});
    // One bright texel per 2x2 block; the average must be a quarter as bright.
    fb.colorAt(0, 0) = Vec3{1.0f};
    fb.colorAt(2, 2) = Vec3{1.0f};

    const Framebuffer resolved = fb.downsample(2);
    CHECK(resolved.width() == 2 && resolved.height() == 2);
    CHECK_NEAR(resolved.colorAt(0, 0).x, 0.25, 1e-6);
    CHECK_NEAR(resolved.colorAt(1, 1).x, 0.25, 1e-6);
    CHECK_NEAR(resolved.colorAt(1, 0).x, 0.0, 1e-6);
}

TEST(srgb_transfer_round_trips) {
    for (float v : {0.0f, 0.002f, 0.05f, 0.25f, 0.5f, 1.0f})
        CHECK_NEAR(srgbToLinear(linearToSrgb(v)), v, 1e-5);
    // Mid grey must brighten on encode, which is the whole point of doing it.
    CHECK(linearToSrgb(0.2f) > 0.4f);
}

TEST(obj_loader_reads_faces_and_attributes) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "sr_obj_loader_test.obj";
    {
        std::ofstream out(path);
        out << "# a single quad\n"
               "v 0 0 0\n"
               "v 1 0 0\n"
               "v 1 1 0\n"
               "v 0 1 0\n"
               "vt 0 0\n"
               "vt 1 0\n"
               "vt 1 1\n"
               "vt 0 1\n"
               "vn 0 0 1\n"
               "f 1/1/1 2/2/1 3/3/1 4/4/1\n";
    }

    std::string error;
    const auto mesh = Mesh::loadObj(path.string(), &error);
    CHECK(mesh.has_value());
    if (mesh) {
        // Four distinct index triples, fan-triangulated into two triangles.
        CHECK(mesh->vertices.size() == 4);
        CHECK(mesh->triangleCount() == 2);
        CHECK_NEAR(mesh->vertices[1].position.x, 1.0, 1e-6);
        CHECK_NEAR(mesh->vertices[1].uv.x, 1.0, 1e-6);
        CHECK_NEAR(mesh->vertices[2].normal.z, 1.0, 1e-6);
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    // A missing file reports failure rather than throwing.
    std::string missingError;
    CHECK(!Mesh::loadObj("definitely-not-here.obj", &missingError).has_value());
    CHECK(!missingError.empty());
}
