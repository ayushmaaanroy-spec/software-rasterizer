# Software Rasterizer

A 3D renderer that runs entirely on the CPU. No OpenGL, Vulkan or DirectX. The
only dependency is the standard library, which includes the PNG writer.

![Showcase render](docs/showcase.png)

960x540, 2x supersampled. 12,302 triangles in 101 ms.

## Build

Needs CMake 3.20+ and a C++20 compiler.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

```bash
./build/srdemo --scene showcase
```

Output goes to `out/`.

## Usage

```
srdemo [options]
  --scene <name>   showcase | normals | clipping | stress | model
  --obj <path>     OBJ file for --scene model
  --width  <n>     output width in pixels        (default 960)
  --height <n>     output height in pixels       (default 540)
  --ss <n>         supersampling factor, 1 = off (default 2)
  --threads <n>    worker threads, 0 = auto      (default 0)
  --frames <n>     render an n-frame turntable   (default 1)
  --out <dir>      output directory              (default out)
  --format <fmt>   png | bmp | ppm               (default png)
  --depth          write the depth buffer instead of shaded color
  --bench          run the threading benchmark and exit
```

Each render reports what the pipeline did:

```
  triangles  submitted 12302 | clipped 2 | culled 6482 | rejected 0 | rasterized 5566
  fragments  covered 1870654 | shaded 1870654 | written 1870654 (0.90x overdraw)
  time       101.4 ms (18.4 Mpix/s coverage)
```

## Pipeline

Geometry, then binning, then raster.

- **Clipping** is Sutherland-Hodgman against seven half-spaces, done in clip
  space before the perspective divide. After the divide, anything crossing the
  eye plane wraps around the screen, since those vertices have `w <= 0`.
- **Culling** keys off the sign of the screen-space area. Front faces are CCW in
  NDC, which the viewport y flip makes negative.
- **Depth** interpolates with plain barycentrics, which is exact because window z
  is affine in screen space. Varyings are not, and need `1/w` correction.
- **Fill rule** is top-left, so a pixel center on a shared edge lands in exactly
  one triangle.
- **Shading** is linear. sRGB is applied once, on write.
- **Shadows** come from a second pass with `colorWrite` off. The bias is in world
  units and divided by the light frustum's depth range at sample time, so
  resizing the frustum does not silently change what it means.

Raster is threaded over 64x64 tiles. One worker owns a tile, so no two threads
touch the same pixel, and the output is bit-identical at any thread count.

Shaders are compile-time polymorphic. Anything satisfying the `ShaderProgram`
concept works, and `Varyings` only has to be a vector space over float.

## Scenes

| | |
|---|---|
| ![](docs/clipping.png) | `clipping` works the clipper. The rails start behind the eye, cross the near plane, span the view and run past the far plane. |
| ![](docs/normals.png) | `normals` draws world-space normals. Fastest way to spot a winding or normal-matrix bug. |
| ![](docs/stress.png) | `stress` is 405 spheres in one 317k-triangle draw call. Used by `--bench`. |

## Performance

16 cores, GCC 14.2 `-O3`, `stress` scene, best of three:

| Threads | 960x540 | speedup | 1920x1080 | speedup |
|--------:|--------:|--------:|----------:|--------:|
| 1  | 79.4 ms | 1.00x | 190.5 ms | 1.00x |
| 2  | 50.8 ms | 1.56x | 116.1 ms | 1.64x |
| 4  | 31.6 ms | 2.51x |  63.9 ms | 2.98x |
| 8  | 23.5 ms | 3.37x |  43.7 ms | 4.36x |
| 16 | 20.9 ms | 3.79x |  34.0 ms | 5.61x |

Same triangles, better scaling at higher resolution, which points at the serial
work between the parallel stages: merging the per-thread geometry output and
binning it. That is the next thing to fix.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

28 tests, aimed at what is easy to get subtly wrong: matrix inverse round-trips,
the normal matrix under non-uniform scale, clipper output satisfying every
plane, a full-screen quad covering every pixel exactly once, perspective-correct
interpolation against an analytic value, and thread-count invariance.

## Layout

```
include/sr/
  math.hpp         Vec2/3/4, Mat4, projection, look-at, inverse
  camera.hpp       view/projection, orbit helper
  mesh.hpp         indexed meshes, primitives, OBJ loading
  texture.hpp      nearest/bilinear sampling, procedural patterns
  framebuffer.hpp  color + depth target, sRGB, image output
  png.hpp          PNG encoder
  clip.hpp         clip-space clipping
  pipeline.hpp     geometry, binning, raster
  shadow.hpp       shadow map fitting, bias, PCF
  shaders.hpp      Blinn-Phong, depth-only, normals, unlit
  thread_pool.hpp  workers shared across draw calls
src/               implementations and the demo driver
tests/             28 tests and a 40 line harness
```

## Known limits

No mipmapping, so minified textures alias and `--ss` is the only antialiasing.
One shadow map, no cascades. No blending sort. The fragment shader is scalar,
not SIMD across a 2x2 quad. DEFLATE uses fixed Huffman codes and greedy
matching, so files run 25-40% larger than zlib's.
