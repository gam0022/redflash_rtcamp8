#include <optixu/optixu_math_namespace.h>
#include "redflash.h"
#include "random.h"
#include <optix_world.h>

using namespace optix;

rtDeclareVariable(float, time, , );

rtDeclareVariable(float, scene_epsilon, , );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

rtDeclareVariable(float3, center, , );
rtDeclareVariable(float3, aabb_min, , );
rtDeclareVariable(float3, aabb_max, , );
rtDeclareVariable(float3, texcoord, attribute texcoord, );

// プライマリレイのDepthを利用した高速化用
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );

float dMenger(float3 z0, float3 offset, float scale) {
    float4 z = make_float4(z0, 1.0);
    for (int n = 0; n < 3; n++) {
        // z = abs(z);
        z.x = abs(z.x);
        z.y = abs(z.y);
        z.z = abs(z.z);
        z.w = abs(z.w);

        // if (z.x < z.y) z.xy = z.yx;
        if (z.x < z.y)
        {
            float x = z.x;
            z.x = z.y;
            z.y = x;
        }

        // if (z.x < z.z) z.xz = z.zx;
        if (z.x < z.z)
        {
            float x = z.x;
            z.x = z.z;
            z.z = x;
        }

        // if (z.y < z.z) z.yz = z.zy;
        if (z.y < z.z)
        {
            float y = z.y;
            z.y = z.z;
            z.z = y;
        }

        z *= scale;
        // z.xyz -= offset * (scale - 1.0);
        z.x -= offset.x * (scale - 1.0);
        z.y -= offset.y * (scale - 1.0);
        z.z -= offset.z * (scale - 1.0);

        if (z.z < -0.5 * offset.z * (scale - 1.0))
            z.z += offset.z * (scale - 1.0);
    }
    // return (length(max(abs(z.xyz) - make_float3(1.0, 1.0, 1.0), 0.0)) - 0.05) / z.w;
    return (length(make_float3(max(abs(z.x) - 1.0, 0.0), max(abs(z.y) - 1.0, 0.0), max(abs(z.z) - 1.0, 0.0))) - 0.05) / z.w;
}

float3 get_xyz(float4 p)
{
    return make_float3(p.x, p.y, p.z);
}

// not work...
void set_xyz(float4& a, float3 b)
{
    a.x = b.x;
    a.y = b.y;
    a.x = b.z;
}

float dMandelFast(float3 p, float scale, int n) {
    float4 q0 = make_float4(p, 1.);
    float4 q = q0;

    for (int i = 0; i < n; i++) {
        // q.xyz = clamp(q.xyz, -1.0, 1.0) * 2.0 - q.xyz;
        // set_xyz(q, clamp(get_xyz(q), -1.0, 1.0) * 2.0 - get_xyz(q));
        float4 tmp = clamp(q, -1.0, 1.0) * 2.0 - q;
        q.x = tmp.x;
        q.y = tmp.y;
        q.z = tmp.z;

        // q = q * scale / clamp( dot( q.xyz, q.xyz ), 0.3, 1.0 ) + q0;
        float3 q_xyz = get_xyz(q);
        q = q * scale / clamp(dot(q_xyz, q_xyz), 0.3, 1.0) + q0;
    }

    // return length( q.xyz ) / abs( q.w );
    return length(get_xyz(q)) / abs(q.w);
}

float fracf(float x)
{
    return x - floor(x);
}

float mod(float a, float b)
{
    return fracf(abs(a / b)) * abs(b);
}

float opRep(float p, float interval)
{
    return mod(p, interval) - interval * 0.5;
}

float map(float3 p)
{
    // return dMenger((p - center) / scale, make_float3(1.23, 1.65, 1.45), 2.56) * scale;
    // return dMenger((p - center) / scale, make_float3(1, 1, 1), 3.1) * scale;

    // dMengerは負荷が高い
    // p.z = opRep(p.z, 20.0);

    if (time < 7)
    {
        float t = 0.0f;
        if (time >= 5.0f)
        {
            t = time - 5;
        }

        float scale = 70.0f;
        return dMandelFast((p - center) / scale, 2.76 + 0.01 * t, 20) * scale;
    }
    else
    {
        float scale = 100;
        return dMenger((p - center) / scale, make_float3(1.2, 1.0, 1.2 + 0.6 * sin(time + 6)), 2.8 + 0.1 * sin(time + 3)) * scale;
    }
}

#define calcNormal(p, dFunc, eps) normalize(\
    make_float3( eps, -eps, -eps) * dFunc(p + make_float3( eps, -eps, -eps)) + \
    make_float3(-eps, -eps,  eps) * dFunc(p + make_float3(-eps, -eps,  eps)) + \
    make_float3(-eps,  eps, -eps) * dFunc(p + make_float3(-eps,  eps, -eps)) + \
    make_float3( eps,  eps,  eps) * dFunc(p + make_float3( eps,  eps,  eps)))

float3 calcNormalBasic(float3 p, float eps)
{
    return normalize(make_float3(
        map(p + make_float3(eps, 0.0, 0.0)) - map(p + make_float3(-eps, 0.0, 0.0)),
        map(p + make_float3(0.0, eps, 0.0)) - map(p + make_float3(0.0, -eps, 0.0)),
        map(p + make_float3(0.0, 0.0, eps)) - map(p + make_float3(0.0, 0.0, -eps))
    ));
}

// https://www.shadertoy.com/view/lttGDn
float calcEdge(float3 p, float width)
{
    float edge = 0.0;
    float2 e = make_float2(width, 0.0f);

    // Take some distance function measurements from either side of the hit point on all three axes.
    float d1 = map(p + make_float3(width, 0.0f, 0.0f)), d2 = map(p - make_float3(width, 0.0f, 0.0f));
    float d3 = map(p + make_float3(0.0f, width, 0.0f)), d4 = map(p - make_float3(0.0f, width, 0.0f));
    float d5 = map(p + make_float3(0.0f, 0.0f, width)), d6 = map(p - make_float3(0.0f, 0.0f, width));
    float d = map(p) * 2.;	// The hit point itself - Doubled to cut down on calculations. See below.

    // Edges - Take a geometry measurement from either side of the hit point. Average them, then see how
    // much the value differs from the hit point itself. Do this for X, Y and Z directions. Here, the sum
    // is used for the overall difference, but there are other ways. Note that it's mainly sharp surface
    // curves that register a discernible difference.
    edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
    //edge = max(max(abs(d1 + d2 - d), abs(d3 + d4 - d)), abs(d5 + d6 - d)); // Etc.

    // Once you have an edge value, it needs to normalized, and smoothed if possible. How you
    // do that is up to you. This is what I came up with for now, but I might tweak it later.
    edge = smoothstep(0., 1., sqrt(edge / e.x * 2.));

    // Return the normal.
    // Standard, normalized gradient mearsurement.
    return edge;
}

RT_CALLABLE_PROGRAM void customMaterialProgram_Nop(MaterialParameter& mat, State& state)
{

}

RT_CALLABLE_PROGRAM void customMaterialProgram_Raymarching(MaterialParameter& mat, State& state)
{
    if (time < 7) return;


    float3 p = state.hitpoint;
    float edge = calcEdge(p, 0.02);
    mat.emission = make_float3(0.2, 0.2, 1) * pow(edge, 2.0f) * abs(sin(0.1 * p.z + 0.5 * time));

    float bar = smoothstep(0.9, 1.0, sin(p.z + 2 * time));
    mat.emission += bar * make_float3(0.2, 0.2, 4.0) * 3;

    mat.roughness = 0.3;
    mat.metallic = 1.0;
    mat.albedo = make_float3(0.6);
}

RT_PROGRAM void intersect(int primIdx)
{
    float eps;
    float t = ray.tmin, d = 0.0;
    float3 p = ray.origin;

    if (current_prd.depth == 0)
    {
        t = max(current_prd.distance, t);
    }

    for (int i = 0; i < 300; i++)
    {
        p = ray.origin + t * ray.direction;
        d = map(p);
        t += d;
        eps = scene_epsilon * t;
        if (abs(d) < eps || t > ray.tmax)
        {
            break;
        }
    }

    if (t < ray.tmax && rtPotentialIntersection(t))
    {
        shading_normal = geometric_normal = calcNormal(p, map, scene_epsilon);
        texcoord = make_float3(p.x, p.y, 0);
        rtReportIntersection(0);
    }
}

RT_PROGRAM void bounds(int, float result[6])
{
    optix::Aabb* aabb = (optix::Aabb*)result;
    aabb->m_min = aabb_min;
    aabb->m_max = aabb_max;
}