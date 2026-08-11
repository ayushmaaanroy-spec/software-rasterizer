#pragma once

#include "sr/math.hpp"

namespace sr {

struct Camera {
    Vec3 eye{0.0f, 0.0f, 5.0f};
    Vec3 target{0.0f, 0.0f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};

    float fovY = radians(50.0f);
    float aspect = 4.0f / 3.0f;
    float zNear = 0.1f;
    float zFar = 100.0f;

    [[nodiscard]] Mat4 view() const noexcept { return lookAt(eye, target, up); }
    [[nodiscard]] Mat4 projection() const noexcept {
        return perspective(fovY, aspect, zNear, zFar);
    }
    [[nodiscard]] Mat4 viewProjection() const noexcept { return projection() * view(); }

    [[nodiscard]] Vec3 forward() const noexcept { return normalize(target - eye); }

    // Place the camera on a sphere around `target`. Yaw sweeps around +Y,
    // pitch lifts towards it.
    [[nodiscard]] static Camera orbit(const Vec3& target, float distance, float yaw, float pitch) {
        Camera cam;
        cam.target = target;
        cam.eye = target + Vec3{distance * std::cos(pitch) * std::sin(yaw),
                                distance * std::sin(pitch),
                                distance * std::cos(pitch) * std::cos(yaw)};
        return cam;
    }
};

}  // namespace sr
