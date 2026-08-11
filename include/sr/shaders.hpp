#pragma once

#include <optional>

#include "sr/math.hpp"
#include "sr/mesh.hpp"
#include "sr/shadow.hpp"
#include "sr/texture.hpp"

namespace sr {

struct DirectionalLight {
    Vec3 direction{-0.5f, -1.0f, -0.4f};  // direction the light travels
    Vec3 color{1.0f, 0.98f, 0.94f};
    float intensity = 1.7f;
};

struct Material {
    Vec3 albedo{0.82f, 0.82f, 0.85f};
    float specular = 0.35f;
    float shininess = 48.0f;
    const Texture* albedoMap = nullptr;
    float uvScale = 1.0f;
    bool useVertexColor = false;
};

struct SurfaceVaryings {
    Vec3 worldPosition;
    Vec3 normal;
    Vec3 color{1.0f, 1.0f, 1.0f};
    Vec2 uv;

    [[nodiscard]] SurfaceVaryings operator*(float s) const noexcept {
        return {worldPosition * s, normal * s, color * s, uv * s};
    }
    [[nodiscard]] SurfaceVaryings operator+(const SurfaceVaryings& o) const noexcept {
        return {worldPosition + o.worldPosition, normal + o.normal, color + o.color, uv + o.uv};
    }
};

// Blinn-Phong, one directional light, hemisphere ambient. All linear; the
// framebuffer applies sRGB on write.
struct LitShader {
    using VertexIn = Vertex;
    using Varyings = SurfaceVaryings;

    Mat4 model = Mat4::identity();
    Mat4 modelNormal = Mat4::identity();
    Mat4 viewProjection = Mat4::identity();
    Vec3 eyePosition;

    DirectionalLight light;
    Material material;
    Vec3 ambientSky{0.16f, 0.19f, 0.26f};
    Vec3 ambientGround{0.05f, 0.045f, 0.04f};
    const ShadowMap* shadowMap = nullptr;

    void setModel(const Mat4& matrix) noexcept {
        model = matrix;
        modelNormal = normalMatrix(matrix);
    }

    [[nodiscard]] Vec4 vertex(const VertexIn& in, Varyings& out) const noexcept {
        const Vec3 world = transformPoint(model, in.position);
        out.worldPosition = world;
        out.normal = transformDirection(modelNormal, in.normal);
        out.color = in.color;
        out.uv = in.uv * material.uvScale;
        return viewProjection * Vec4(world, 1.0f);
    }

    [[nodiscard]] std::optional<Vec4> fragment(const Varyings& in) const noexcept {
        const Vec3 n = normalize(in.normal);

        Vec3 albedo = material.albedo;
        if (material.useVertexColor) albedo = albedo * in.color;
        if (material.albedoMap != nullptr)
            albedo = albedo * material.albedoMap->sampleBilinear(in.uv.x, in.uv.y);

        const Vec3 toLight = normalize(-light.direction);
        const Vec3 toEye = normalize(eyePosition - in.worldPosition);
        const Vec3 halfway = normalize(toLight + toEye);

        const float nDotL = std::max(0.0f, dot(n, toLight));
        const float nDotH = std::max(0.0f, dot(n, halfway));
        const float specular =
            nDotL > 0.0f ? std::pow(nDotH, material.shininess) * material.specular : 0.0f;

        const Vec3 ambient = lerp(ambientGround, ambientSky, n.y * 0.5f + 0.5f);

        // Only the direct term is occluded, so shadowed areas keep their ambient
        // and the edge of the shadow map does not read as a hard line.
        const float visibility =
            shadowMap != nullptr ? shadowMap->visibility(in.worldPosition, n, nDotL) : 1.0f;

        const Vec3 direct = light.color * (light.intensity * visibility);
        const Vec3 radiance = albedo * (ambient + direct * nDotL) + direct * specular;
        return Vec4(radiance, 1.0f);
    }
};

// Position only, for filling a shadow map.
struct DepthOnlyShader {
    using VertexIn = Vertex;

    struct Varyings {
        [[nodiscard]] Varyings operator*(float) const noexcept { return {}; }
        [[nodiscard]] Varyings operator+(const Varyings&) const noexcept { return {}; }
    };

    Mat4 model = Mat4::identity();
    Mat4 viewProjection = Mat4::identity();

    void setModel(const Mat4& matrix) noexcept { model = matrix; }

    [[nodiscard]] Vec4 vertex(const VertexIn& in, Varyings&) const noexcept {
        return viewProjection * (model * Vec4(in.position, 1.0f));
    }

    [[nodiscard]] std::optional<Vec4> fragment(const Varyings&) const noexcept {
        return Vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
};

// World-space normals as color. Quickest way to spot a winding or normal-matrix
// mistake.
struct NormalShader {
    using VertexIn = Vertex;
    using Varyings = SurfaceVaryings;

    Mat4 model = Mat4::identity();
    Mat4 modelNormal = Mat4::identity();
    Mat4 viewProjection = Mat4::identity();

    void setModel(const Mat4& matrix) noexcept {
        model = matrix;
        modelNormal = normalMatrix(matrix);
    }

    [[nodiscard]] Vec4 vertex(const VertexIn& in, Varyings& out) const noexcept {
        const Vec3 world = transformPoint(model, in.position);
        out.worldPosition = world;
        out.normal = transformDirection(modelNormal, in.normal);
        out.color = in.color;
        out.uv = in.uv;
        return viewProjection * Vec4(world, 1.0f);
    }

    [[nodiscard]] std::optional<Vec4> fragment(const Varyings& in) const noexcept {
        const Vec3 n = normalize(in.normal);
        return Vec4(n * 0.5f + Vec3(0.5f), 1.0f);
    }
};

struct UnlitShader {
    using VertexIn = Vertex;
    using Varyings = SurfaceVaryings;

    Mat4 model = Mat4::identity();
    Mat4 viewProjection = Mat4::identity();
    Vec3 tint{1.0f, 1.0f, 1.0f};
    float alpha = 1.0f;
    const Texture* albedoMap = nullptr;
    float uvScale = 1.0f;
    float alphaCutoff = 0.0f;  // > 0 discards texels darker than this

    void setModel(const Mat4& matrix) noexcept { model = matrix; }

    [[nodiscard]] Vec4 vertex(const VertexIn& in, Varyings& out) const noexcept {
        const Vec3 world = transformPoint(model, in.position);
        out.worldPosition = world;
        out.normal = in.normal;
        out.color = in.color;
        out.uv = in.uv * uvScale;
        return viewProjection * Vec4(world, 1.0f);
    }

    [[nodiscard]] std::optional<Vec4> fragment(const Varyings& in) const noexcept {
        Vec3 color = tint * in.color;
        if (albedoMap != nullptr) {
            const Vec3 sampled = albedoMap->sampleBilinear(in.uv.x, in.uv.y);
            if (alphaCutoff > 0.0f && sampled.x < alphaCutoff) return std::nullopt;
            color = color * sampled;
        }
        return Vec4(color, alpha);
    }
};

}  // namespace sr
