// Column-vector convention: v' = M * v. Mat4 storage is row-major, m[row][col].
#pragma once

#include <algorithm>
#include <cmath>

namespace sr {

inline constexpr float kPi = 3.14159265358979323846f;

[[nodiscard]] inline constexpr float radians(float degrees) noexcept {
    return degrees * (kPi / 180.0f);
}

[[nodiscard]] inline constexpr float clampf(float v, float lo, float hi) noexcept {
    return v < lo ? lo : (v > hi ? hi : v);
}

struct Vec2 {
    float x{}, y{};

    constexpr Vec2() = default;
    constexpr Vec2(float x_, float y_) noexcept : x(x_), y(y_) {}

    constexpr Vec2 operator+(const Vec2& o) const noexcept { return {x + o.x, y + o.y}; }
    constexpr Vec2 operator-(const Vec2& o) const noexcept { return {x - o.x, y - o.y}; }
    constexpr Vec2 operator*(float s) const noexcept { return {x * s, y * s}; }
    constexpr Vec2 operator/(float s) const noexcept { return {x / s, y / s}; }
    constexpr Vec2& operator+=(const Vec2& o) noexcept { x += o.x; y += o.y; return *this; }
};

struct Vec3 {
    float x{}, y{}, z{};

    constexpr Vec3() = default;
    constexpr explicit Vec3(float s) noexcept : x(s), y(s), z(s) {}
    constexpr Vec3(float x_, float y_, float z_) noexcept : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator+(const Vec3& o) const noexcept { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const noexcept { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator-() const noexcept { return {-x, -y, -z}; }
    constexpr Vec3 operator*(float s) const noexcept { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(float s) const noexcept { return {x / s, y / s, z / s}; }
    // Component-wise, for modulating colors.
    constexpr Vec3 operator*(const Vec3& o) const noexcept { return {x * o.x, y * o.y, z * o.z}; }

    constexpr Vec3& operator+=(const Vec3& o) noexcept { x += o.x; y += o.y; z += o.z; return *this; }
    constexpr Vec3& operator-=(const Vec3& o) noexcept { x -= o.x; y -= o.y; z -= o.z; return *this; }
    constexpr Vec3& operator*=(float s) noexcept { x *= s; y *= s; z *= s; return *this; }

    constexpr float operator[](int i) const noexcept { return i == 0 ? x : (i == 1 ? y : z); }
};

[[nodiscard]] inline constexpr Vec3 operator*(float s, const Vec3& v) noexcept { return v * s; }

[[nodiscard]] inline constexpr float dot(const Vec3& a, const Vec3& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] inline constexpr Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

[[nodiscard]] inline float length(const Vec3& v) noexcept { return std::sqrt(dot(v, v)); }

[[nodiscard]] inline Vec3 normalize(const Vec3& v) noexcept {
    const float len2 = dot(v, v);
    if (len2 <= 1e-20f) return {0.0f, 0.0f, 0.0f};
    return v * (1.0f / std::sqrt(len2));
}

[[nodiscard]] inline constexpr Vec3 lerp(const Vec3& a, const Vec3& b, float t) noexcept {
    return a * (1.0f - t) + b * t;
}

[[nodiscard]] inline constexpr Vec3 minv(const Vec3& a, const Vec3& b) noexcept {
    return {a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z};
}

[[nodiscard]] inline constexpr Vec3 maxv(const Vec3& a, const Vec3& b) noexcept {
    return {a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z};
}

[[nodiscard]] inline constexpr Vec3 saturate(const Vec3& v) noexcept {
    return {clampf(v.x, 0.0f, 1.0f), clampf(v.y, 0.0f, 1.0f), clampf(v.z, 0.0f, 1.0f)};
}

[[nodiscard]] inline constexpr Vec3 reflect(const Vec3& i, const Vec3& n) noexcept {
    return i - n * (2.0f * dot(i, n));
}

struct Vec4 {
    float x{}, y{}, z{}, w{};

    constexpr Vec4() = default;
    constexpr Vec4(float x_, float y_, float z_, float w_) noexcept : x(x_), y(y_), z(z_), w(w_) {}
    constexpr Vec4(const Vec3& v, float w_) noexcept : x(v.x), y(v.y), z(v.z), w(w_) {}

    [[nodiscard]] constexpr Vec3 xyz() const noexcept { return {x, y, z}; }

    constexpr Vec4 operator+(const Vec4& o) const noexcept { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    constexpr Vec4 operator-(const Vec4& o) const noexcept { return {x - o.x, y - o.y, z - o.z, w - o.w}; }
    constexpr Vec4 operator*(float s) const noexcept { return {x * s, y * s, z * s, w * s}; }

    constexpr float operator[](int i) const noexcept {
        return i == 0 ? x : (i == 1 ? y : (i == 2 ? z : w));
    }
};

[[nodiscard]] inline constexpr float dot(const Vec4& a, const Vec4& b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

struct Mat4 {
    float m[4][4]{};

    [[nodiscard]] static constexpr Mat4 identity() noexcept {
        Mat4 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
        return r;
    }

    [[nodiscard]] constexpr Mat4 operator*(const Mat4& b) const noexcept {
        Mat4 r;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                r.m[i][j] = m[i][0] * b.m[0][j] + m[i][1] * b.m[1][j] +
                            m[i][2] * b.m[2][j] + m[i][3] * b.m[3][j];
            }
        }
        return r;
    }

    [[nodiscard]] constexpr Vec4 operator*(const Vec4& v) const noexcept {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
                m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w};
    }

    [[nodiscard]] constexpr Mat4 transposed() const noexcept {
        Mat4 r;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) r.m[i][j] = m[j][i];
        return r;
    }
};

[[nodiscard]] inline constexpr Vec3 transformPoint(const Mat4& mat, const Vec3& p) noexcept {
    const Vec4 r = mat * Vec4(p, 1.0f);
    return r.xyz();
}

[[nodiscard]] inline constexpr Vec3 transformDirection(const Mat4& mat, const Vec3& d) noexcept {
    const Vec4 r = mat * Vec4(d, 0.0f);
    return r.xyz();
}

[[nodiscard]] inline constexpr Mat4 translation(const Vec3& t) noexcept {
    Mat4 r = Mat4::identity();
    r.m[0][3] = t.x;
    r.m[1][3] = t.y;
    r.m[2][3] = t.z;
    return r;
}

[[nodiscard]] inline constexpr Mat4 scaling(const Vec3& s) noexcept {
    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;
    r.m[1][1] = s.y;
    r.m[2][2] = s.z;
    return r;
}

[[nodiscard]] inline Mat4 rotationX(float angle) noexcept {
    const float c = std::cos(angle), s = std::sin(angle);
    Mat4 r = Mat4::identity();
    r.m[1][1] = c; r.m[1][2] = -s;
    r.m[2][1] = s; r.m[2][2] = c;
    return r;
}

[[nodiscard]] inline Mat4 rotationY(float angle) noexcept {
    const float c = std::cos(angle), s = std::sin(angle);
    Mat4 r = Mat4::identity();
    r.m[0][0] = c;  r.m[0][2] = s;
    r.m[2][0] = -s; r.m[2][2] = c;
    return r;
}

[[nodiscard]] inline Mat4 rotationZ(float angle) noexcept {
    const float c = std::cos(angle), s = std::sin(angle);
    Mat4 r = Mat4::identity();
    r.m[0][0] = c; r.m[0][1] = -s;
    r.m[1][0] = s; r.m[1][1] = c;
    return r;
}

[[nodiscard]] inline Mat4 rotationAxis(const Vec3& axis, float angle) noexcept {
    const Vec3 a = normalize(axis);
    const float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
    Mat4 r = Mat4::identity();
    r.m[0][0] = t * a.x * a.x + c;
    r.m[0][1] = t * a.x * a.y - s * a.z;
    r.m[0][2] = t * a.x * a.z + s * a.y;
    r.m[1][0] = t * a.x * a.y + s * a.z;
    r.m[1][1] = t * a.y * a.y + c;
    r.m[1][2] = t * a.y * a.z - s * a.x;
    r.m[2][0] = t * a.x * a.z - s * a.y;
    r.m[2][1] = t * a.y * a.z + s * a.x;
    r.m[2][2] = t * a.z * a.z + c;
    return r;
}

// Right-handed: the camera looks down -Z in eye space.
[[nodiscard]] inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) noexcept {
    const Vec3 f = normalize(center - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;  r.m[0][1] = s.y;  r.m[0][2] = s.z;  r.m[0][3] = -dot(s, eye);
    r.m[1][0] = u.x;  r.m[1][1] = u.y;  r.m[1][2] = u.z;  r.m[1][3] = -dot(u, eye);
    r.m[2][0] = -f.x; r.m[2][1] = -f.y; r.m[2][2] = -f.z; r.m[2][3] = dot(f, eye);
    return r;
}

// Maps z into [-1, 1], the OpenGL convention.
[[nodiscard]] inline Mat4 perspective(float fovY, float aspect, float zNear, float zFar) noexcept {
    const float f = 1.0f / std::tan(fovY * 0.5f);
    Mat4 r;
    r.m[0][0] = f / aspect;
    r.m[1][1] = f;
    r.m[2][2] = (zFar + zNear) / (zNear - zFar);
    r.m[2][3] = (2.0f * zFar * zNear) / (zNear - zFar);
    r.m[3][2] = -1.0f;
    return r;
}

[[nodiscard]] inline constexpr Mat4 orthographic(float l, float r_, float b, float t,
                                                 float zNear, float zFar) noexcept {
    Mat4 r = Mat4::identity();
    r.m[0][0] = 2.0f / (r_ - l);
    r.m[1][1] = 2.0f / (t - b);
    r.m[2][2] = -2.0f / (zFar - zNear);
    r.m[0][3] = -(r_ + l) / (r_ - l);
    r.m[1][3] = -(t + b) / (t - b);
    r.m[2][3] = -(zFar + zNear) / (zFar - zNear);
    return r;
}

// Cofactor expansion. Returns identity if the matrix is singular.
[[nodiscard]] inline Mat4 inverse(const Mat4& mat) noexcept {
    const float* a = &mat.m[0][0];
    float inv[16];

    inv[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2]  =  a[1]*a[6]*a[15]  - a[1]*a[7]*a[14]  - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7]  - a[13]*a[3]*a[6];
    inv[6]  = -a[0]*a[6]*a[15]  + a[0]*a[7]*a[14]  + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7]  + a[12]*a[3]*a[6];
    inv[10] =  a[0]*a[5]*a[15]  - a[0]*a[7]*a[13]  - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7]  - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14]  + a[0]*a[6]*a[13]  + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6]  + a[12]*a[2]*a[5];
    inv[3]  = -a[1]*a[6]*a[11]  + a[1]*a[7]*a[10]  + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7]   + a[9]*a[3]*a[6];
    inv[7]  =  a[0]*a[6]*a[11]  - a[0]*a[7]*a[10]  - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7]   - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11]  + a[0]*a[7]*a[9]   + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7]   + a[8]*a[3]*a[5];
    inv[15] =  a[0]*a[5]*a[10]  - a[0]*a[6]*a[9]   - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6]   - a[8]*a[2]*a[5];

    float det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    if (std::fabs(det) < 1e-20f) return Mat4::identity();

    det = 1.0f / det;
    Mat4 out;
    float* o = &out.m[0][0];
    for (int i = 0; i < 16; ++i) o[i] = inv[i] * det;
    return out;
}

// Normals need this instead of the model matrix under non-uniform scale.
[[nodiscard]] inline Mat4 normalMatrix(const Mat4& model) noexcept {
    return inverse(model).transposed();
}

}  // namespace sr
