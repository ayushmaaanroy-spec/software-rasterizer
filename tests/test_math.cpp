#include "check.hpp"
#include "sr/camera.hpp"
#include "sr/math.hpp"
#include "sr/mesh.hpp"

using namespace sr;

namespace {

Mat4 sampleTransform() {
    return translation({1.5f, -2.0f, 3.25f}) * rotationAxis({0.3f, 1.0f, -0.4f}, radians(37.0f)) *
           scaling({2.0f, 0.5f, 1.75f});
}

}  // namespace

TEST(vector_algebra) {
    const Vec3 a{1.0f, 2.0f, 3.0f};
    const Vec3 b{-4.0f, 5.0f, 6.0f};

    CHECK_NEAR(dot(a, b), -4.0 + 10.0 + 18.0, 1e-6);

    // Cross product is perpendicular to both inputs and right-handed.
    const Vec3 c = cross(a, b);
    CHECK_NEAR(dot(c, a), 0.0, 1e-5);
    CHECK_NEAR(dot(c, b), 0.0, 1e-5);
    const Vec3 z = cross(Vec3{1, 0, 0}, Vec3{0, 1, 0});
    CHECK_NEAR(z.z, 1.0, 1e-6);

    CHECK_NEAR(length(normalize(b)), 1.0, 1e-6);
    // Degenerate input must not produce NaN.
    const Vec3 zero = normalize(Vec3{0.0f});
    CHECK(zero.x == 0.0f && zero.y == 0.0f && zero.z == 0.0f);
}

TEST(matrix_identity_and_product) {
    const Mat4 identity = Mat4::identity();
    const Vec4 v{1.0f, -2.0f, 3.0f, 1.0f};
    const Vec4 r = identity * v;
    CHECK_NEAR(r.x, 1.0, 1e-6);
    CHECK_NEAR(r.y, -2.0, 1e-6);
    CHECK_NEAR(r.z, 3.0, 1e-6);
    CHECK_NEAR(r.w, 1.0, 1e-6);

    // Matrix product must compose in the same order as applying the transforms.
    const Mat4 t = translation({1.0f, 2.0f, 3.0f});
    const Mat4 s = scaling({2.0f, 2.0f, 2.0f});
    const Vec3 composed = transformPoint(t * s, Vec3{1.0f, 1.0f, 1.0f});
    CHECK_NEAR(composed.x, 3.0, 1e-6);
    CHECK_NEAR(composed.y, 4.0, 1e-6);
    CHECK_NEAR(composed.z, 5.0, 1e-6);
}

TEST(matrix_inverse_round_trips) {
    const Mat4 m = sampleTransform();
    const Mat4 product = m * inverse(m);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) CHECK_NEAR(product.m[i][j], i == j ? 1.0 : 0.0, 1e-4);

    // Singular input falls back to the identity instead of producing NaNs.
    Mat4 singular{};
    const Mat4 fallback = inverse(singular);
    CHECK_NEAR(fallback.m[0][0], 1.0, 1e-6);
}

TEST(rotation_matrices) {
    // A quarter turn about +Y sends +X to -Z (right-handed).
    const Vec3 rotated = transformDirection(rotationY(radians(90.0f)), Vec3{1.0f, 0.0f, 0.0f});
    CHECK_NEAR(rotated.x, 0.0, 1e-6);
    CHECK_NEAR(rotated.z, -1.0, 1e-6);

    // rotationAxis about Y must agree with the dedicated rotationY.
    const Mat4 byAxis = rotationAxis({0.0f, 1.0f, 0.0f}, radians(33.0f));
    const Mat4 direct = rotationY(radians(33.0f));
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) CHECK_NEAR(byAxis.m[i][j], direct.m[i][j], 1e-5);

    // Rotations preserve length.
    CHECK_NEAR(length(transformDirection(byAxis, Vec3{1.0f, 2.0f, -3.0f})),
               length(Vec3{1.0f, 2.0f, -3.0f}), 1e-5);
}

TEST(normal_matrix_handles_non_uniform_scale) {
    // Under a non-uniform scale the naive transform tilts normals off the
    // surface; the inverse-transpose keeps them perpendicular to tangents.
    const Mat4 model = scaling({3.0f, 0.25f, 1.0f});
    const Vec3 tangent{1.0f, 1.0f, 0.0f};
    const Vec3 normal{-1.0f, 1.0f, 0.0f};
    CHECK_NEAR(dot(tangent, normal), 0.0, 1e-6);

    const Vec3 newTangent = transformDirection(model, tangent);
    const Vec3 newNormal = transformDirection(normalMatrix(model), normal);
    CHECK_NEAR(dot(normalize(newTangent), normalize(newNormal)), 0.0, 1e-5);
}

TEST(look_at_builds_a_view_basis) {
    const Vec3 eye{3.0f, 4.0f, 5.0f};
    const Vec3 target{0.0f, 1.0f, 0.0f};
    const Mat4 view = lookAt(eye, target, Vec3{0.0f, 1.0f, 0.0f});

    // The eye maps to the origin of view space...
    const Vec3 eyeInView = transformPoint(view, eye);
    CHECK_NEAR(length(eyeInView), 0.0, 1e-4);

    // ...and the target sits straight down -Z at the right distance.
    const Vec3 targetInView = transformPoint(view, target);
    CHECK_NEAR(targetInView.x, 0.0, 1e-4);
    CHECK_NEAR(targetInView.y, 0.0, 1e-4);
    CHECK_NEAR(targetInView.z, -length(target - eye), 1e-4);
}

TEST(perspective_maps_the_frustum_to_the_clip_cube) {
    const float zNear = 0.5f, zFar = 40.0f;
    const Mat4 proj = perspective(radians(90.0f), 1.0f, zNear, zFar);

    // A point on the near plane lands at z = -1 after the perspective divide.
    const Vec4 nearPoint = proj * Vec4(Vec3{0.0f, 0.0f, -zNear}, 1.0f);
    CHECK_NEAR(nearPoint.w, zNear, 1e-4);
    CHECK_NEAR(nearPoint.z / nearPoint.w, -1.0, 1e-4);

    // ...and one on the far plane at z = +1.
    const Vec4 farPoint = proj * Vec4(Vec3{0.0f, 0.0f, -zFar}, 1.0f);
    CHECK_NEAR(farPoint.z / farPoint.w, 1.0, 1e-4);

    // With a 90 degree vertical field of view, y = -z sits on the top edge.
    const Vec4 edge = proj * Vec4(Vec3{0.0f, 4.0f, -4.0f}, 1.0f);
    CHECK_NEAR(edge.y / edge.w, 1.0, 1e-4);

    // Points behind the eye come out with w < 0, which is exactly what the
    // clipper keys on.
    const Vec4 behind = proj * Vec4(Vec3{0.0f, 0.0f, 1.0f}, 1.0f);
    CHECK(behind.w < 0.0f);
}

TEST(camera_orbit_keeps_its_radius) {
    const Vec3 target{1.0f, 2.0f, -3.0f};
    const Camera cam = Camera::orbit(target, 6.0f, radians(35.0f), radians(20.0f));
    CHECK_NEAR(length(cam.eye - target), 6.0, 1e-4);
    CHECK_NEAR(length(cam.forward()), 1.0, 1e-5);
}

TEST(primitive_meshes_are_well_formed) {
    const Mesh cube = Mesh::cube(2.0f);
    CHECK(cube.vertices.size() == 24);
    CHECK(cube.triangleCount() == 12);

    const Bounds b = cube.bounds();
    CHECK_NEAR(b.min.x, -1.0, 1e-6);
    CHECK_NEAR(b.max.y, 1.0, 1e-6);

    // Every cube normal points away from the centre.
    for (const Vertex& v : cube.vertices) CHECK(dot(v.position, v.normal) > 0.0f);

    const Mesh sphere = Mesh::uvSphere(1.0f, 8, 16);
    CHECK(sphere.triangleCount() == 8 * 16 * 2);
    for (const Vertex& v : sphere.vertices) {
        CHECK_NEAR(length(v.position), 1.0, 1e-5);
        // On a unit sphere the normal is the position.
        CHECK_NEAR(dot(normalize(v.position), v.normal), 1.0, 1e-5);
    }

    const Mesh torus = Mesh::torus(1.0f, 0.3f, 12, 8);
    CHECK(torus.triangleCount() == 12 * 8 * 2);
    for (const Vertex& v : torus.vertices) CHECK_NEAR(length(v.normal), 1.0, 1e-5);
}

TEST(recomputed_normals_match_analytic_ones) {
    Mesh sphere = Mesh::uvSphere(1.0f, 24, 48);
    const std::vector<Vertex> analytic = sphere.vertices;
    sphere.recomputeNormals();

    // Area-weighted vertex normals on a dense sphere should be very close to
    // the exact ones. The seam column only sees half its neighbourhood, so
    // allow a little slack -- but every normal must still be unit length and
    // point outwards, including the pole caps where the faces degenerate.
    double worst = 0.0;
    bool allUnitLength = true;
    for (std::size_t i = 0; i < sphere.vertices.size(); ++i) {
        if (std::fabs(length(sphere.vertices[i].normal) - 1.0f) > 1e-4f) allUnitLength = false;
        worst = std::max<double>(worst, 1.0 - dot(sphere.vertices[i].normal, analytic[i].normal));
    }
    CHECK(allUnitLength);
    CHECK(worst < 0.02);
}

TEST(mesh_transform_and_normalize) {
    Mesh cube = Mesh::cube(1.0f);
    cube.transform(translation({5.0f, 0.0f, 0.0f}) * scaling({4.0f, 1.0f, 1.0f}));

    const Bounds b = cube.bounds();
    CHECK_NEAR(b.center().x, 5.0, 1e-5);
    CHECK_NEAR(b.extent().x, 4.0, 1e-5);
    // Non-uniform scale must not leave normals unnormalised.
    for (const Vertex& v : cube.vertices) CHECK_NEAR(length(v.normal), 1.0, 1e-5);

    cube.normalizeToUnitSize(2.0f);
    const Bounds n = cube.bounds();
    CHECK_NEAR(n.extent().x, 2.0, 1e-5);
    CHECK_NEAR(n.center().x, 0.0, 1e-5);
}
