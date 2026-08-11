# Software Rasterizer

A CPU 3D rendering pipeline written from scratch in C++20. Every stage between a
vertex buffer and a pixel is implemented here — vertex transforms, homogeneous
frustum clipping, the perspective divide, triangle scan conversion, depth
buffering and shading. There is no OpenGL, Vulkan, DirectX or any other external
library; the only dependency is the C++ standard library. That extends to the
image files: the PNG encoder, including its DEFLATE compressor, is in `src/png.cpp`.

![Showcase render](docs/showcase.png)

*960×540, 2× supersampled. Blinn-Phong shading, procedural textures, shadow
mapping, depth-buffered occlusion. 12,302 triangles in 101 ms on the CPU.*

## The pipeline

A draw call moves through three stages.

**Geometry.** Each vertex runs through the shader's `vertex()` function, which
returns a clip-space position and fills in whatever varyings the fragment stage
needs. Triangles are then clipped, culled and mapped to pixels:

- **Clipping** happens in homogeneous clip space, *before* the perspective
  divide, using Sutherland–Hodgman against seven half-spaces (`w > ε` plus the
  six frustum planes). Doing it here is what stops geometry crossing the eye
  plane from wrapping around the screen: those vertices have `w ≤ 0`, and
  dividing by `w` would fold them back into view. Clipping a triangle can turn
  it into a polygon of up to 9 vertices, which is fan-triangulated.
- **Back-face culling** uses the sign of the screen-space signed area. Front
  faces are counter-clockwise in NDC, which the y-flip in the viewport transform
  turns into a negative area.
- **The viewport transform** divides by `w`, maps NDC to pixels and to a window
  depth in `[0, 1]`. `1/w` is kept per-vertex for the raster stage.

**Binning.** Every surviving triangle is filed into each 64×64 tile its bounding
box touches.

**Raster.** Tiles are claimed by worker threads and scan-converted. Coverage
comes from the three edge functions evaluated at each pixel centre, stepped
incrementally along a scanline. Inside a covered pixel:

- **Depth** is interpolated with the plain screen-space barycentrics — window-space
  z is affine in screen space, so that is exact — and tested before the fragment
  shader runs.
- **Varyings are perspective-correct**: `1/w` is interpolated linearly, then each
  vertex is reweighted by its own `1/w`. Interpolating attributes directly in
  screen space is the classic wrong answer, and on a receding ground plane it is
  off by a factor of three at the midpoint. There is a test pinning this down.
- **The fill rule** is top-left, so a pixel centre landing exactly on a shared
  edge belongs to exactly one of the two triangles. Without it, seams are either
  double-shaded or dropped.

Shading is done in linear space; the sRGB transfer curve is applied once, when
the framebuffer is written to disk.

## Shadows

Shadow mapping needs no new machinery: the scene is drawn a second time from the
light with `colorWrite` off, and the depth buffer that comes out is a record of
the nearest occluder in every direction. Shading then re-projects each point into
that map and compares.

The interesting part is the bias. Compare a surface against a depth map it is
itself in and it shadows itself, in stripes. Three things keep that in check: a
constant bias for depth quantisation, a slope-scaled term that grows as the
surface turns edge-on to the light, and a normal-offset that moves the lookup off
the surface before projecting. All three are specified in **world units** and
divided by the light frustum's depth range at sample time — expressed in
normalised depth they silently change meaning whenever the light frustum is
resized, and a value tuned on one scene produces acne on the next. 3×3
percentage-closer filtering softens the edge.

The map covers only a bounding sphere of the casters, not the whole scene;
receivers outside it are treated as lit, which is what lets a 34-unit ground
plane take shadows from a 12-unit map. Being depth-only, it skips allocating a
colour buffer — at 2048² that would be 50 MB of nothing.

## Threading

The raster stage is parallel over tiles, and a tile is owned by exactly one
worker, so no two threads ever touch the same pixel of the colour or depth
buffer. There are no atomics on the framebuffer and no locks in the inner loop.

Because the tile grid is a fixed decomposition of the target — it does not depend
on the thread count — and because bin order is global submission order, output is
**bit-identical at any thread count**. That is asserted by a test, not assumed.

The worker threads live in a pool that outlives individual draw calls. This
matters more than it sounds like it should: a scene of 35 small draw calls
spawning threads per draw spent more time creating threads than rasterizing, and
scaled 1.25× on 16 cores. With the pool it scales 4.2×.

## Building

Requires CMake 3.20+ and a C++20 compiler.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
```

Then:

```bash
./build/srdemo --scene showcase
```

Renders are written to `out/` as PNG.

## Usage

```
srdemo [options]
  --scene <name>   showcase | normals | clipping | stress | model
  --obj <path>     OBJ file to load for --scene model
  --width  <n>     output width in pixels        (default 960)
  --height <n>     output height in pixels       (default 540)
  --ss <n>         supersampling factor, 1 = off (default 2)
  --threads <n>    worker threads, 0 = auto      (default 0)
  --frames <n>     render an n-frame turntable   (default 1)
  --out <dir>      output directory              (default out)
  --format <fmt>   png | bmp | ppm               (default png)
  --depth          write the depth buffer instead of shaded colour
  --bench          run the threading benchmark and exit
```

Each render reports what the pipeline actually did:

```
  triangles  submitted 12302 | clipped 2 | culled 6482 | rejected 0 | rasterized 5566
  fragments  covered 1870654 | shaded 1870654 | written 1870654 (0.90x overdraw)
  time       101.4 ms (18.4 Mpix/s coverage)
```

(Counts are for the camera pass; the time includes the shadow pass.)

## Scenes

| | |
|---|---|
| ![](docs/clipping.png) | **clipping** — a corridor built to exercise the clipper. The rails start behind the eye, cross the near plane, run the full width of the view and carry on past the far plane. |
| ![](docs/normals.png) | **normals** — interpolated world-space normals, the fastest way to spot a winding or normal-matrix mistake. |
| ![](docs/stress.png) | **stress** — 405 spheres merged into one 317k-triangle draw call, used by `--bench`. |

## Performance

16-core CPU, GCC 14.2 `-O3`, `stress` scene (317,520 triangles), best of three runs:

| Threads | 960×540 | speedup | 1920×1080 | speedup |
|--------:|--------:|--------:|----------:|--------:|
| 1  | 79.4 ms | 1.00× | 190.5 ms | 1.00× |
| 2  | 50.8 ms | 1.56× | 116.1 ms | 1.64× |
| 4  | 31.6 ms | 2.51× |  63.9 ms | 2.98× |
| 8  | 23.5 ms | 3.37× |  43.7 ms | 4.36× |
| 16 | 20.9 ms | 3.79× |  34.0 ms | 5.61× |

Scaling is better at the higher resolution with the *same* triangle count, which
locates the remaining bottleneck: the per-triangle work between the two parallel
stages — merging the per-thread geometry output and binning it into tiles — is
still serial. Raising the fragment count raises the parallel fraction and the
speedup goes with it. Parallelising the bin build is the obvious next step.

## Layout

```
include/sr/
  math.hpp         Vec2/3/4, Mat4, projection, look-at, inverse, normal matrix
  camera.hpp       view/projection matrices, orbit helper
  mesh.hpp         indexed triangle meshes, primitives, OBJ loading
  texture.hpp      texture sampling (nearest/bilinear) and procedural patterns
  framebuffer.hpp  colour + depth target, sRGB encode, image output
  png.hpp          PNG encoder: CRC-32, Adler-32 and a DEFLATE compressor
  clip.hpp         Sutherland-Hodgman clipping in homogeneous clip space
  pipeline.hpp     the rasterizer: geometry, binning and raster stages
  shadow.hpp       shadow map fitting, bias and percentage-closer filtering
  shaders.hpp      Blinn-Phong, depth-only, normal-visualisation, unlit
  thread_pool.hpp  persistent workers shared across draw calls
src/               non-templated implementations, plus the demo driver
tests/             28 tests and a ~40 line harness
```

Shaders are compile-time polymorphic. A shader is any type satisfying the
`ShaderProgram` concept — it names a `VertexIn` and a `Varyings` type and
provides `vertex()` and `fragment()` — so the fragment function inlines straight
into the raster inner loop instead of going through a virtual call per pixel.
The `Varyings` type only has to be a vector space over `float`; clipping and
perspective-correct interpolation are written against `operator*(float)` and
`operator+`, so a shader can carry whatever it likes across a triangle.

## Tests

```bash
cmake --build build && ctest --test-dir build --output-on-failure
```

28 tests covering the parts that are easy to get subtly wrong: matrix inverse
round-trips, the normal matrix under non-uniform scale, `perspective()` mapping
the frustum onto the clip cube, clipper output satisfying every plane, a
screen-filling quad covering every pixel *exactly once* (which is what pins down
the fill rule), perspective-correct interpolation against an analytic value,
depth-test ordering, fragment discard, and thread-count invariance of the
rendered image.

The PNG tests walk the encoded stream and validate every chunk length and CRC,
the zlib header, the deflate block type and an independently recomputed
Adler-32 — because a project with nothing to decode PNG with cannot round-trip
its own output. That the bytes are also acceptable to a real decoder was checked
separately against Pillow, which reproduces the BMP path's pixels exactly.

## Limitations

No mipmapping, so minified textures alias — supersampling (`--ss`) is the only
antialiasing on offer. Shadows come from a single map with no cascades, so
coverage is traded against resolution. No blending sort, and the fragment shader
is scalar rather than SIMD across a 2×2 quad. The DEFLATE compressor uses fixed
Huffman codes and greedy matching, so its files run 25–40% larger than zlib's.
