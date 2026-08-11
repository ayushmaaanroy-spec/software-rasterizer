// Demo driver: builds a few scenes, renders them, writes them out.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "sr/camera.hpp"
#include "sr/framebuffer.hpp"
#include "sr/mesh.hpp"
#include "sr/pipeline.hpp"
#include "sr/shaders.hpp"
#include "sr/texture.hpp"

using namespace sr;

namespace {

struct Options {
    std::string scene = "showcase";
    std::string outDir = "out";
    std::string objPath;
    std::string format = "png";
    int width = 960;
    int height = 540;
    int supersample = 2;
    int threads = 0;
    int frames = 1;
    bool depthView = false;
    bool benchmark = false;
};

struct Assets {
    Texture checker;
    Texture grid;
    Texture rust;

    Assets() {
        checker = Texture::checker(256, 8, Vec3{0.90f, 0.90f, 0.92f}, Vec3{0.10f, 0.12f, 0.17f});
        grid = Texture::grid(512, 16, Vec3{0.16f, 0.17f, 0.20f}, Vec3{0.55f, 0.56f, 0.60f}, 3);
        rust = Texture::noise(256, Vec3{0.24f, 0.10f, 0.05f}, Vec3{0.85f, 0.48f, 0.20f});
    }
};

using Clock = std::chrono::steady_clock;

double millisSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void fillSky(Framebuffer& fb, const Vec3& top, const Vec3& bottom) {
    const int h = fb.height();
    for (int y = 0; y < h; ++y) {
        const float t = h > 1 ? static_cast<float>(y) / static_cast<float>(h - 1) : 0.0f;
        const Vec3 color = lerp(top, bottom, t * t);
        for (int x = 0; x < fb.width(); ++x) fb.colorAt(x, y) = color;
    }
}

template <ShaderProgram S>
void drawMesh(Rasterizer& raster, const S& shader, const Mesh& mesh, const PipelineConfig& config) {
    raster.draw(shader, mesh.vertices, mesh.indices, config);
}

DirectionalLight keyLight() {
    DirectionalLight light;
    light.direction = normalize(Vec3{-0.55f, -0.85f, -0.40f});
    light.color = Vec3{1.0f, 0.97f, 0.90f};
    light.intensity = 1.9f;
    return light;
}

// Shadowed scenes submit their geometry twice, once from the light and once
// from the camera, so the object list is built up front rather than drawn inline.
struct SceneObject {
    const Mesh* mesh = nullptr;
    Mat4 model = Mat4::identity();
    Material material;
    bool castsShadow = true;
};

// Depth-only pass from the light's point of view.
void renderShadowMap(ShadowMap& shadowMap, const std::vector<SceneObject>& objects,
                     const PipelineConfig& config) {
    Rasterizer raster(shadowMap.target());

    PipelineConfig shadowConfig = config;
    shadowConfig.colorWrite = false;

    DepthOnlyShader shader;
    shader.viewProjection = shadowMap.lightViewProjection();

    for (const SceneObject& object : objects) {
        if (!object.castsShadow) continue;
        shader.setModel(object.model);
        drawMesh(raster, shader, *object.mesh, shadowConfig);
    }
}

void renderLit(Rasterizer& raster, const Camera& camera, const DirectionalLight& light,
               const ShadowMap* shadowMap, const std::vector<SceneObject>& objects,
               const PipelineConfig& config) {
    LitShader shader;
    shader.viewProjection = camera.viewProjection();
    shader.eyePosition = camera.eye;
    shader.light = light;
    shader.shadowMap = shadowMap;

    for (const SceneObject& object : objects) {
        shader.setModel(object.model);
        shader.material = object.material;
        drawMesh(raster, shader, *object.mesh, config);
    }
}


void sceneShowcase(Rasterizer& raster, const PipelineConfig& config, const Assets& assets,
                   float time) {
    const Camera camera = [&] {
        Camera c = Camera::orbit(Vec3{0.0f, 0.95f, 0.0f}, 8.4f, time * 0.9f, radians(21.0f));
        c.aspect = raster.target().aspect();
        c.fovY = radians(45.0f);
        c.zNear = 0.1f;
        c.zFar = 90.0f;
        return c;
    }();

    // Must outlive `objects`, which only stores pointers.
    const Mesh ground = Mesh::plane(34.0f, 1);
    const Mesh cube = Mesh::cube(1.5f);
    const Mesh sphere = Mesh::uvSphere(1.0f, 32, 64);
    const Mesh torus = Mesh::torus(0.85f, 0.28f, 64, 32);

    std::vector<SceneObject> objects;

    // Big enough that its far edge stays out of frame. The grid is coarse on
    // purpose, there is no mip chain to filter it. Receives shadows but does not
    // cast, so it stays out of the map and leaves the resolution to the casters.
    SceneObject& floor = objects.emplace_back();
    floor.mesh = &ground;
    floor.castsShadow = false;
    floor.material.albedo = Vec3{0.62f, 0.63f, 0.66f};
    floor.material.albedoMap = &assets.grid;
    floor.material.uvScale = 6.0f;
    floor.material.specular = 0.10f;
    floor.material.shininess = 24.0f;

    // Textured cube, spinning about a tilted axis and floating clear of the floor.
    SceneObject& box = objects.emplace_back();
    box.mesh = &cube;
    box.model = translation(Vec3{-2.45f, 1.45f, 0.20f}) *
                rotationAxis(Vec3{0.25f, 1.0f, 0.15f}, time * 1.3f);
    box.material.albedo = Vec3{1.0f, 1.0f, 1.0f};
    box.material.albedoMap = &assets.checker;
    box.material.uvScale = 0.5f;
    box.material.specular = 0.45f;
    box.material.shininess = 64.0f;

    // Smooth, glossy sphere.
    SceneObject& redSphere = objects.emplace_back();
    redSphere.mesh = &sphere;
    redSphere.model = translation(Vec3{0.30f, 0.95f, -0.55f}) * scaling(Vec3{0.95f});
    redSphere.material.albedo = Vec3{0.85f, 0.28f, 0.22f};
    redSphere.material.specular = 0.75f;
    redSphere.material.shininess = 128.0f;

    // Textured sphere.
    SceneObject& rustSphere = objects.emplace_back();
    rustSphere.mesh = &sphere;
    rustSphere.model =
        translation(Vec3{2.55f, 0.62f, 0.95f}) * scaling(Vec3{0.62f}) * rotationY(time * 0.6f);
    rustSphere.material.albedo = Vec3{1.0f, 1.0f, 1.0f};
    rustSphere.material.albedoMap = &assets.rust;
    rustSphere.material.specular = 0.20f;
    rustSphere.material.shininess = 20.0f;

    // Tilted just far enough to show the hole. At this angle its lowest point
    // sits a hair above y = 0.
    SceneObject& ring = objects.emplace_back();
    ring.mesh = &torus;
    ring.model = translation(Vec3{0.75f, 0.88f, 2.75f}) * rotationY(time * 1.1f) *
                 rotationX(radians(32.0f));
    ring.material.albedo = Vec3{0.28f, 0.55f, 0.85f};
    ring.material.specular = 0.60f;
    ring.material.shininess = 96.0f;

    const DirectionalLight light = keyLight();

    // Only needs to cover the casters and where their shadows land, not the
    // whole ground plane. Receivers outside it count as lit.
    ShadowMap shadowMap(2048, light.direction, Vec3{0.0f, 0.9f, 0.6f}, 6.0f);
    renderShadowMap(shadowMap, objects, config);

    renderLit(raster, camera, light, &shadowMap, objects, config);
}

void sceneNormals(Rasterizer& raster, const PipelineConfig& config, const Assets&, float time) {
    Camera camera = Camera::orbit(Vec3{0.0f, 0.0f, 0.0f}, 4.6f, time * 0.8f, radians(22.0f));
    camera.aspect = raster.target().aspect();
    camera.fovY = radians(45.0f);

    NormalShader shader;
    shader.viewProjection = camera.viewProjection();

    const Mesh torus = Mesh::torus(1.15f, 0.45f, 96, 48);
    shader.setModel(rotationX(radians(28.0f)) * rotationZ(time * 0.7f));
    drawMesh(raster, shader, torus, config);
}

// Built to work the clipper: geometry crosses the near plane, runs past the far
// plane, and spills off all four sides.
void sceneClipping(Rasterizer& raster, const PipelineConfig& config, const Assets& assets,
                   float time) {
    Camera camera;
    camera.eye = Vec3{std::sin(time * 0.7f) * 1.6f, 1.2f, 4.0f};
    camera.target = Vec3{0.0f, 1.0f, -12.0f};
    camera.aspect = raster.target().aspect();
    camera.fovY = radians(60.0f);
    camera.zNear = 0.25f;
    camera.zFar = 26.0f;  // deliberately short, so the corridor gets far-clipped

    LitShader shader;
    shader.viewProjection = camera.viewProjection();
    shader.eyePosition = camera.eye;
    shader.light = keyLight();
    shader.light.intensity = 1.5f;

    // Room turned inside out with the camera in it, so every side plane has to
    // clip these walls.
    Mesh room = Mesh::cube(26.0f);
    room.flipWinding();
    shader.setModel(translation(Vec3{0.0f, 6.0f, -8.0f}));
    shader.material = Material{};
    shader.material.albedo = Vec3{0.55f, 0.57f, 0.62f};
    shader.material.albedoMap = &assets.grid;
    shader.material.uvScale = 2.0f;  // coarse on purpose: no mip chain to filter with
    shader.material.specular = 0.05f;
    drawMesh(raster, shader, room, config);

    // Columns receding into the distance. The nearest ones start behind the eye.
    const Mesh column = Mesh::cube(1.0f);
    for (int i = 0; i < 16; ++i) {
        const float z = 5.5f - static_cast<float>(i) * 3.1f;
        for (int side = -1; side <= 1; side += 2) {
            shader.setModel(translation(Vec3{static_cast<float>(side) * 2.4f, 1.6f, z}) *
                            scaling(Vec3{1.0f, 3.2f, 1.0f}));
            shader.material = Material{};
            shader.material.albedo = lerp(Vec3{0.90f, 0.35f, 0.25f}, Vec3{0.25f, 0.55f, 0.90f},
                                          static_cast<float>(i) / 15.0f);
            shader.material.specular = 0.40f;
            shader.material.shininess = 64.0f;
            drawMesh(raster, shader, column, config);
        }
    }

    // One rail hits everything at once. It starts behind the eye, crosses the
    // near plane, runs the full width of the view and carries on past the far
    // plane.
    for (int side = -1; side <= 1; side += 2) {
        shader.setModel(translation(Vec3{static_cast<float>(side) * 2.4f, 0.22f, -17.5f}) *
                        scaling(Vec3{0.5f, 0.45f, 45.0f}));
        shader.material = Material{};
        shader.material.albedo = Vec3{0.95f, 0.78f, 0.22f};
        shader.material.specular = 0.55f;
        shader.material.shininess = 80.0f;
        drawMesh(raster, shader, column, config);
    }
}

// A dense lattice of spheres, merged into one draw call. Used for benchmarking.
Mesh buildStressMesh() {
    const Mesh unitSphere = Mesh::uvSphere(0.42f, 14, 28);

    Mesh merged;
    for (int x = 0; x < 9; ++x) {
        for (int y = 0; y < 5; ++y) {
            for (int z = 0; z < 9; ++z) {
                Mesh instance = unitSphere;
                instance.transform(translation(Vec3{static_cast<float>(x - 4) * 1.05f,
                                                    static_cast<float>(y - 2) * 1.05f,
                                                    static_cast<float>(z - 4) * 1.05f}));
                instance.setColor(Vec3{static_cast<float>(x) / 8.0f,
                                       static_cast<float>(y) / 4.0f,
                                       static_cast<float>(z) / 8.0f});
                merged.append(instance);
            }
        }
    }
    return merged;
}

void sceneStress(Rasterizer& raster, const PipelineConfig& config, const Assets&, float time,
                 const Mesh& mesh) {
    Camera camera = Camera::orbit(Vec3{0.0f, 0.0f, 0.0f}, 15.0f, time * 0.5f, radians(18.0f));
    camera.aspect = raster.target().aspect();
    camera.fovY = radians(45.0f);
    camera.zFar = 80.0f;

    LitShader shader;
    shader.viewProjection = camera.viewProjection();
    shader.eyePosition = camera.eye;
    shader.light = keyLight();
    shader.setModel(Mat4::identity());
    shader.material = Material{};
    shader.material.albedo = Vec3{1.0f, 1.0f, 1.0f};
    shader.material.useVertexColor = true;
    shader.material.specular = 0.5f;
    shader.material.shininess = 48.0f;
    drawMesh(raster, shader, mesh, config);
}

void sceneModel(Rasterizer& raster, const PipelineConfig& config, const Assets&, float time,
                const Mesh& mesh) {
    Camera camera = Camera::orbit(Vec3{0.0f, 0.15f, 0.0f}, 4.2f, time * 0.9f, radians(14.0f));
    camera.aspect = raster.target().aspect();
    camera.fovY = radians(45.0f);

    Mesh ground = Mesh::plane(12.0f, 1);
    ground.transform(translation(Vec3{0.0f, -1.05f, 0.0f}));

    std::vector<SceneObject> objects;

    SceneObject& floor = objects.emplace_back();
    floor.mesh = &ground;
    floor.castsShadow = false;
    floor.material.albedo = Vec3{0.35f, 0.36f, 0.40f};
    floor.material.specular = 0.05f;

    SceneObject& model = objects.emplace_back();
    model.mesh = &mesh;
    model.material.albedo = Vec3{0.80f, 0.76f, 0.70f};
    model.material.specular = 0.35f;
    model.material.shininess = 40.0f;

    const DirectionalLight light = keyLight();
    ShadowMap shadowMap(2048, light.direction, Vec3{0.0f, -0.2f, 0.0f}, 2.6f);
    renderShadowMap(shadowMap, objects, config);

    renderLit(raster, camera, light, &shadowMap, objects, config);
}


void printStats(const RenderStats& s, double ms, int pixels) {
    std::printf("  triangles  submitted %llu | clipped %llu | culled %llu | rejected %llu | rasterized %llu\n",
                static_cast<unsigned long long>(s.trianglesSubmitted),
                static_cast<unsigned long long>(s.trianglesClipped),
                static_cast<unsigned long long>(s.trianglesCulled),
                static_cast<unsigned long long>(s.trianglesRejected),
                static_cast<unsigned long long>(s.trianglesRasterized));
    std::printf("  fragments  covered %llu | shaded %llu | written %llu (%.2fx overdraw)\n",
                static_cast<unsigned long long>(s.fragmentsCovered),
                static_cast<unsigned long long>(s.fragmentsShaded),
                static_cast<unsigned long long>(s.fragmentsWritten),
                pixels > 0 ? static_cast<double>(s.fragmentsCovered) / pixels : 0.0);
    std::printf("  time       %.1f ms (%.1f Mpix/s coverage)\n", ms,
                ms > 0.0 ? static_cast<double>(s.fragmentsCovered) / (ms * 1000.0) : 0.0);
}

bool renderFrame(const Options& opt, const Assets& assets, const Mesh& extraMesh, float time,
                 const std::string& outPath, bool verbose) {
    Framebuffer fb(opt.width * opt.supersample, opt.height * opt.supersample);
    fillSky(fb, Vec3{0.055f, 0.070f, 0.105f}, Vec3{0.180f, 0.200f, 0.240f});
    fb.clearDepth();

    Rasterizer raster(fb);
    PipelineConfig config;
    config.threads = opt.threads;

    const Clock::time_point start = Clock::now();
    if (opt.scene == "showcase") {
        sceneShowcase(raster, config, assets, time);
    } else if (opt.scene == "normals") {
        sceneNormals(raster, config, assets, time);
    } else if (opt.scene == "clipping") {
        sceneClipping(raster, config, assets, time);
    } else if (opt.scene == "stress") {
        sceneStress(raster, config, assets, time, extraMesh);
    } else if (opt.scene == "model") {
        sceneModel(raster, config, assets, time, extraMesh);
    } else {
        std::fprintf(stderr, "unknown scene '%s'\n", opt.scene.c_str());
        return false;
    }
    const double ms = millisSince(start);

    Framebuffer resolved = fb.downsample(opt.supersample);
    if (opt.depthView) resolved.visualizeDepth();

    const bool written = opt.format == "png"   ? resolved.writePNG(outPath)
                         : opt.format == "bmp" ? resolved.writeBMP(outPath)
                         : opt.format == "ppm" ? resolved.writePPM(outPath)
                                               : false;
    if (!written) {
        std::fprintf(stderr, "failed to write '%s'\n", outPath.c_str());
        return false;
    }
    if (verbose) {
        std::printf("wrote %s (%dx%d, %dx supersampled)\n", outPath.c_str(), opt.width, opt.height,
                    opt.supersample);
        printStats(raster.stats(), ms, fb.width() * fb.height());
    }
    return true;
}

void runBenchmark(const Options& opt, const Assets& assets, const Mesh& stressMesh) {
    Options bench = opt;
    bench.scene = "stress";
    bench.supersample = 1;

    std::printf("benchmark: %s at %dx%d, %zu triangles\n\n", bench.scene.c_str(), bench.width,
                bench.height, stressMesh.triangleCount());

    double baseline = 0.0;
    for (int threads : {1, 2, 4, 8, 16}) {
        if (threads > 1 && static_cast<unsigned>(threads / 2) >=
                               std::max(1u, std::thread::hardware_concurrency()))
            break;

        Framebuffer fb(bench.width, bench.height);
        Rasterizer raster(fb);
        PipelineConfig config;
        config.threads = threads;

        // One untimed warm-up pass, then three timed passes; report the best.
        double best = 0.0;
        for (int pass = 0; pass < 4; ++pass) {
            fillSky(fb, Vec3{0.05f, 0.06f, 0.09f}, Vec3{0.16f, 0.18f, 0.22f});
            fb.clearDepth();
            raster.resetStats();

            const Clock::time_point start = Clock::now();
            sceneStress(raster, config, assets, 0.6f, stressMesh);
            const double ms = millisSince(start);
            if (pass > 0 && (best == 0.0 || ms < best)) best = ms;
        }

        if (threads == 1) baseline = best;
        std::printf("  %2d thread(s): %7.1f ms   speedup %4.2fx\n", threads, best,
                    baseline > 0.0 ? baseline / best : 1.0);
    }
}

void printUsage() {
    std::printf(
        "software rasterizer demo\n"
        "\n"
        "usage: srdemo [options]\n"
        "  --scene <name>   showcase | normals | clipping | stress | model  (default showcase)\n"
        "  --obj <path>     OBJ file for --scene model\n"
        "  --width  <n>     output width in pixels        (default 960)\n"
        "  --height <n>     output height in pixels       (default 540)\n"
        "  --ss <n>         supersampling factor, 1 = off (default 2)\n"
        "  --threads <n>    worker threads, 0 = auto      (default 0)\n"
        "  --frames <n>     render an n-frame turntable   (default 1)\n"
        "  --out <dir>      output directory              (default out)\n"
        "  --format <fmt>   png | bmp | ppm               (default png)\n"
        "  --depth          write the depth buffer instead of shaded color\n"
        "  --bench          run the threading benchmark and exit\n"
        "  --help\n");
}

bool parseArgs(int argc, char** argv, Options& opt) {
    auto needsValue = [&](int i) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            return false;
        }
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        } else if (arg == "--depth") {
            opt.depthView = true;
        } else if (arg == "--bench") {
            opt.benchmark = true;
        } else if (arg == "--scene" && needsValue(i)) {
            opt.scene = argv[++i];
        } else if (arg == "--obj" && needsValue(i)) {
            opt.objPath = argv[++i];
        } else if (arg == "--out" && needsValue(i)) {
            opt.outDir = argv[++i];
        } else if (arg == "--format" && needsValue(i)) {
            opt.format = argv[++i];
        } else if (arg == "--width" && needsValue(i)) {
            opt.width = std::atoi(argv[++i]);
        } else if (arg == "--height" && needsValue(i)) {
            opt.height = std::atoi(argv[++i]);
        } else if (arg == "--ss" && needsValue(i)) {
            opt.supersample = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--threads" && needsValue(i)) {
            opt.threads = std::atoi(argv[++i]);
        } else if (arg == "--frames" && needsValue(i)) {
            opt.frames = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "unrecognized option '%s'\n\n", arg.c_str());
            printUsage();
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parseArgs(argc, argv, opt)) return 1;

    const Assets assets;

    Mesh extraMesh;
    if (opt.scene == "stress" || opt.benchmark) {
        extraMesh = buildStressMesh();
    } else if (opt.scene == "model") {
        std::string error;
        auto loaded = Mesh::loadObj(opt.objPath, &error);
        if (!loaded) {
            std::fprintf(stderr, "model scene: %s\n", error.c_str());
            return 1;
        }
        extraMesh = std::move(*loaded);
        extraMesh.normalizeToUnitSize(2.0f);
        std::printf("loaded %s: %zu vertices, %zu triangles\n", opt.objPath.c_str(),
                    extraMesh.vertices.size(), extraMesh.triangleCount());
    }

    if (opt.benchmark) {
        runBenchmark(opt, assets, extraMesh);
        return 0;
    }

    std::error_code ec;
    std::filesystem::create_directories(opt.outDir, ec);

    if (opt.format != "png" && opt.format != "bmp" && opt.format != "ppm") {
        std::fprintf(stderr, "unknown format '%s' (expected png, bmp or ppm)\n", opt.format.c_str());
        return 1;
    }

    if (opt.frames == 1) {
        const std::string path = opt.outDir + "/" + opt.scene + "." + opt.format;
        return renderFrame(opt, assets, extraMesh, 0.6f, path, true) ? 0 : 1;
    }

    const Clock::time_point start = Clock::now();
    for (int frame = 0; frame < opt.frames; ++frame) {
        char name[256];
        std::snprintf(name, sizeof(name), "%s/%s_%03d.%s", opt.outDir.c_str(), opt.scene.c_str(),
                      frame, opt.format.c_str());
        // One full revolution across the frame range.
        const float time = 2.0f * kPi * static_cast<float>(frame) / static_cast<float>(opt.frames);
        if (!renderFrame(opt, assets, extraMesh, time, name, false)) return 1;
        std::printf("\rframe %d/%d", frame + 1, opt.frames);
        std::fflush(stdout);
    }
    std::printf("\rrendered %d frames in %.1f s -> %s/\n", opt.frames, millisSince(start) / 1000.0,
                opt.outDir.c_str());
    return 0;
}
