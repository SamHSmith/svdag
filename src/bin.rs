use antelope::camera::RenderCamera;
use antelope::cgmath::prelude::One;
use antelope::cgmath::{Deg, Euler, InnerSpace, Matrix4, Quaternion, Vector3, Vector4};
use antelope::mesh::{Mesh, MeshCreateInfo, MeshFactory, PostVertex, RenderInfo, Vertex};
use antelope::window::{DemoTriangleRenderer, Frame, TriangleFrame, Window};
use antelope::{MeshFrame, MeshRenderer};
use lib::*;

const EPSILON: f64 = 0.000002;

fn cast_ray_v_box(start: Vector3<f64>, dir: Vector3<f64>, size: f64) -> (f64, Vector3<f64>) {
    let t1 = (size * -0.5 - start.x) / dir.x;
    let t2 = (size * 0.5 - start.x) / dir.x;

    let t3 = (size * -0.5 - start.y) / dir.y;
    let t4 = (size * 0.5 - start.y) / dir.y;

    let t5 = (size * -0.5 - start.z) / dir.z;
    let t6 = (size * 0.5 - start.z) / dir.z;

    #[inline(always)]
    fn inbox(x: f64, y: f64, z: f64, size: f64) -> bool {
        return x + EPSILON >= size * -0.5
            && x - EPSILON <= size * 0.5
            && y + EPSILON >= size * -0.5
            && y - EPSILON <= size * 0.5
            && z + EPSILON >= size * -0.5
            && z - EPSILON <= size * 0.5;
    }

    let pos1 = start + dir * t1;
    let pos2 = start + dir * t2;
    let pos3 = start + dir * t3;
    let pos4 = start + dir * t4;
    let pos5 = start + dir * t5;
    let pos6 = start + dir * t6;

    let mut b1 =
        (1.0 - (inbox(pos1.x, pos1.y, pos1.z, size) && t1 > 0.0) as i64 as f64) * std::f64::MAX;
    let mut b2 =
        (1.0 - (inbox(pos2.x, pos2.y, pos2.z, size) && t2 > 0.0) as i64 as f64) * std::f64::MAX;
    let mut b3 =
        (1.0 - (inbox(pos3.x, pos3.y, pos3.z, size) && t3 > 0.0) as i64 as f64) * std::f64::MAX;
    let mut b4 =
        (1.0 - (inbox(pos4.x, pos4.y, pos4.z, size) && t4 > 0.0) as i64 as f64) * std::f64::MAX;
    let mut b5 =
        (1.0 - (inbox(pos5.x, pos5.y, pos5.z, size) && t5 > 0.0) as i64 as f64) * std::f64::MAX;
    let mut b6 =
        (1.0 - (inbox(pos6.x, pos6.y, pos6.z, size) && t6 > 0.0) as i64 as f64) * std::f64::MAX;

    b1 = b1.max(t1);
    b2 = b2.max(t2);
    b3 = b3.max(t3);
    b4 = b4.max(t4);
    b5 = b5.max(t5);
    b6 = b6.max(t6);

    let min = b1.min(b2.min(b3.min(b4.min(b5.min(b6)))));

    (
        min,
        (pos1 * (b1 == min) as u64 as f64)
            + (pos2 * (b2 == min) as u64 as f64)
            + (pos3 * (b3 == min) as u64 as f64)
            + (pos4 * (b4 == min) as u64 as f64)
            + (pos5 * (b5 == min) as u64 as f64)
            + (pos6 * (b6 == min) as u64 as f64),
    )
}

struct RayTarget<'a> {
    distance: f64,
    node: &'a VoxelNode,
    nodelocation: Vector3<f64>,
    hitpos: Vector3<f64>, //relative to the omnicube
}

struct RayResult {
    colour: Vector3<f64>,
    emission: f64,
    metalness: f64,
    roughness: f64,
    hitlocation: Vector3<f64>,
    hitnormal: Vector3<f64>,
    dead: usize,
}

fn dead_ray() -> RayResult {
    RayResult {
        colour: new_vec(0.0, 0.0, 0.0),
        emission: 0.0,
        metalness: 0.0,
        roughness: 0.0,
        hitlocation: new_vec(0.0, 0.0, 0.0),
        hitnormal: new_vec(0.0, 0.0, 0.0),
        dead: 0,
    }
}

fn cast_ray_voxel<'a>(
    start: Vector3<f64>,
    dir: Vector3<f64>,
    root: &VoxelNode,
    tree: VoxelTree,
) -> RayResult {
    let mut size: f64 = 1.0;
    let mut activenodes: Vec<&VoxelNode> = Vec::new();
    let mut nodelocations: Vec<Vector3<f64>> = Vec::new();
    activenodes.push(root);
    nodelocations.push(new_vec(0.0, 0.0, 0.0));

    let mut closestleaf = (std::f64::MAX, None);

    let mut x = 0;
    while x < activenodes.len() {
        let mut targets: Vec<RayTarget> = Vec::new();
        for i in 0..activenodes.len() {
            let (distance, pos) = cast_ray_v_box(start - nodelocations[i], dir, size);
            targets.push(RayTarget {
                distance,
                node: activenodes[i],
                nodelocation: nodelocations[i],
                hitpos: pos,
            });

            x += 1;
        }
        targets.sort_by(|a, b| (a.distance).partial_cmp(&b.distance).unwrap());
        for h in 0..targets.len() {
            let hit = &targets[h];
            if (hit.distance.is_normal() && hit.distance < std::f64::MAX) {
                if hit.node.flags & 1 != 0 {
                    //leaf
                    if hit.distance < closestleaf.0 {
                        closestleaf = (
                            hit.distance,
                            Some(RayResult {
                                colour: new_vec(
                                    hit.node.colour[0] as f64,
                                    hit.node.colour[1] as f64,
                                    hit.node.colour[2] as f64,
                                ) / 255.0,
                                emission: hit.node.emission as f64 / 255.0,
                                metalness: hit.node.metalness as f64 / 255.0,
                                roughness: hit.node.roughness as f64 / 255.0,
                                hitlocation: hit.nodelocation + hit.hitpos,
                                hitnormal: get_biggest_axis(hit.hitpos).normalize(),
                                dead: 1,
                            }),
                        );
                    }
                } else {
                    for c in 0..8 {
                        let child: u8 = 1 << c;

                        if hit.node.childmask & child != 0 {
                            nodelocations
                                .push(hit.nodelocation + (oct_byte_to_vec(child) * size / 2.0));
                            activenodes.push(hit.node.get_child(tree, c));
                        }
                    }
                }
            }
        }

        activenodes.drain(0..x);
        nodelocations.drain(0..x);
        x = 0;
        size /= 2.0;
    }

    match closestleaf.1 {
        Some(c) => return c,
        None => return dead_ray(),
    }
}

const RPP: usize
    = 20;
const RBC: usize = 4;
const EMISSION: f64 = 15.0;

fn cast_pixel(
    fxmin: f64,
    fxmax: f64,
    fymin: f64,
    fymax: f64,
    campos: Vector3<f64>,
    camrot: Quaternion<f64>,
    tree: VoxelTree,
    root: &VoxelNode,
) -> [u8; 3] {
    fn do_ray(
        dir: Vector3<f64>,
        start: Vector3<f64>,
        bouncesleft: usize,
        tree: VoxelTree,
        root: &VoxelNode,
    ) -> Vector3<f64> {
        if (bouncesleft <= 0) {
            return new_vec(0.0, 0.0, 0.0);
        }

        let cast1 = cast_ray_voxel(start, dir, root, tree);

        if cast1.dead == 0 {
            return new_vec(0.0, 0.0, 0.0);
        }

        let fresnel = fresnel(-dir, cast1.hitnormal) * (1.0 - cast1.roughness);

        let specular = fresnel * ((0.5 * (1.0 - cast1.metalness)) + (0.98 * cast1.metalness));

        let specular_highlight = ((new_vec(0.5, 0.5, 0.5) * (1.0 - cast1.metalness))
            + (cast1.colour * cast1.metalness))
            * specular;

        let diffuse = 0.5 * (1.0 - cast1.metalness);

        let newdir = rand_dir_from_surf(cast1.hitnormal);

        let diffuse_colour = (1.0 - cast1.emission)
            * (diffuse
                * point_wise_mul(
                    cast1.colour,
                    do_ray(newdir, cast1.hitlocation, bouncesleft - 1, tree, root),
                ));
/*
        let specular_colour = (1.0 - cast1.emission)
            * point_wise_mul(
                specular_highlight,
                do_ray(
                    (newdir * cast1.roughness)
                        + (reflect(dir, cast1.hitnormal) * (1.0 - cast1.roughness)),
                    cast1.hitlocation,
                    bouncesleft - 1,
                    tree,
                    root,
                ),
            );
*/
        let emissive_colour = (cast1.colour * cast1.emission * EMISSION);

        emissive_colour + diffuse_colour// + specular_colour
    }

    let mut colour: Vector3<f64> = new_vec(0.0, 0.0, 0.0);
    let mut count = 0;

    for _ in 0..RPP {
        let raydir = camrot
            * new_vec(
                (random_mix(fxmin, fxmax) * 2.0 - 1.0),
                (random_mix(fymin, fymax) * 2.0 - 1.0),
                1.0,
            );
        count += 1;
        colour += do_ray(raydir, campos, RBC, tree, root);
    }

    colour /= count as f64;

    return linear_to_srgb(colour);
}

fn main() {
    let mut tree = allocate(500);
    let node: &mut VoxelNode = tree.allocate_and_get_node();
    node.put_child(1, tree.allocate_node());
    node.get_child(tree, 1).flags = 1;
    node.get_child(tree, 1).colour = [70, 50, 90];
    node.get_child(tree, 1).roughness = 210;
    node.put_child(0, tree.allocate_node());
    let node2: &mut VoxelNode = node.get_child(tree, 0);
    node2.put_child(3, tree.allocate_node());
    node2.get_child(tree, 3).flags = 1;
    node2.get_child(tree, 3).colour = [130, 100, 2];
    node2.get_child(tree, 3).roughness = 35;
    node2.put_child(2, tree.allocate_node());
    node2.get_child(tree, 2).flags = 1;
    node2.get_child(tree, 2).colour = [255, 219, 145];
    node2.get_child(tree, 2).roughness = 20;
    node2.get_child(tree, 2).metalness = 255;
    node2.put_child(5, tree.allocate_node());
    node2.get_child(tree, 5).flags = 1;
    node2.get_child(tree, 5).colour = [100, 100, 100];
    node2.get_child(tree, 5).emission = 255;
    node2.get_child(tree, 5).roughness = 255;
    node2.put_child(7, tree.allocate_node());
    node2.get_child(tree, 7).flags = 1;
    node2.get_child(tree, 7).colour = [20, 50, 180];
    node2.get_child(tree, 7).roughness = 240;

    //node.flags =1;
    node.colour = [255, 183, 235];

    let width: usize = 256;
    let height: usize = 256;

    let campos = new_vec(-0.7, -0.8, -1.8);
    let camrot = new_vec(-20.0, 20.0, 0.0);
    let rotation = Quaternion::from(Euler {
        x: Deg(camrot.x),
        y: Deg(camrot.y),
        z: Deg(camrot.z),
    });

    let cubepos = new_vec(0.0, 0.0, 0.0);

    let mut buffer: Vec<u8> = vec![0; (width * height * 3) as usize]; // Generate the image data;

    let startcpu = std::time::Instant::now();

    use rayon::prelude::*;
    use std::sync::Mutex;

    let columnsdone = Mutex::new(0 as usize);

    let maxindex : usize = width * height;
    let chunksize : usize = 1000;

    (0..(maxindex / chunksize)).into_par_iter().for_each(|w| {
        for s in 0..(chunksize.min(maxindex.saturating_sub(w*chunksize))) {
            let x = ((w * chunksize) + s) % width;
            let y = (((w * chunksize) + s) - x) / width;
            let pixelwidth = 1.0 / width as f64;
            let fx = x as f64 * pixelwidth;
            let pixelheight = 1.0 / height as f64;
            let fy = y as f64 * pixelheight;

            let hw = pixelwidth / 2.0;
            let hh = pixelheight / 2.0;

            let index = (((w * chunksize)+s) * 3 as usize);
            let colour: [u8; 3] = cast_pixel(
                fx - hw,
                fx + hw,
                fy - hh,
                fy + hh,
                campos,
                rotation,
                tree,
                node,
            );
            let bufferptr = buffer.as_ptr();
            unsafe {
                *(bufferptr.offset(index as isize) as *mut u8) = colour[0];
                *(bufferptr.offset((index + 1) as isize) as *mut u8) = colour[1];
                *(bufferptr.offset((index + 2) as isize) as *mut u8) = colour[2];
            }
        }
        let mut percent = columnsdone.lock().unwrap();
        *percent += 1;
        println!(
            "Cpu Render is {} % done",
            ((*percent * chunksize as usize) as f64 * 100.0) / maxindex as f64
        );
    });

    let cpuend = std::time::Instant::now();

    // Save the buffer as "image.png"
    image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width as u32, height as u32, &buffer[..])
        .unwrap()
        .save("image.png")
        .unwrap();

    println!("Cpu took {} ms", (cpuend - startcpu).as_millis());

    use image::ImageBuffer;
    use image::Rgba;
    use std::sync::Arc;
    use vulkano::buffer::BufferUsage;
    use vulkano::buffer::CpuAccessibleBuffer;
    use vulkano::command_buffer::AutoCommandBufferBuilder;
    use vulkano::command_buffer::CommandBuffer;
    use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
    use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
    use vulkano::device::Device;
    use vulkano::device::DeviceExtensions;
    use vulkano::device::Features;
    use vulkano::format::Format;
    use vulkano::image::Dimensions;
    use vulkano::image::StorageImage;
    use vulkano::instance::Instance;
    use vulkano::instance::InstanceExtensions;
    use vulkano::instance::PhysicalDevice;
    use vulkano::pipeline::ComputePipeline;
    use vulkano::sync::GpuFuture;

    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");

    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");

    let mut feat = Features::none();
    feat.shader_f3264 = true;

    let mut exts = DeviceExtensions::none();
    exts.khr_storage_buffer_storage_class = true;

    let (device, mut queues) = {
        Device::new(
            physical,
            &feat,
            &exts,
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let image = StorageImage::new(
        device.clone(),
        Dimensions::Dim2d { width:width as u32, height:height as u32},
        Format::R8G8B8A8Unorm,
        Some(queue.family()),
    )
    .unwrap();

    let vbuffer = unsafe {
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::uniform_buffer(),
            false,
            (0..500)
                .into_iter()
                .map(|i| *(tree.base as *const VoxelNode as *const u32).offset(i)),
        )
        .unwrap()
    };

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
#version 450
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308
#define VBUFFER_SIZE 500

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
layout(set = 1, binding = 0) uniform VoxelBuffer {
    vec4[VBUFFER_SIZE] data;
};

layout(push_constant) uniform pushConstants {
    dvec3 campos;
    dvec3 right;
    dvec3 up;
    dvec3 forward;
} pc;

float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

vec3 reflectvec(vec3 v, vec3 normal){
    return v + (-2.0 * dot(v, normal) * normal);
}

vec3 rand_dir_from_surf(vec3 normal, vec2 co){
    vec3 v = vec3(rand(co), rand(co * 2), rand(co *3));

    if(dot(v, normal) < 0.0) {
        v += (-2.0 * dot(v, normal) * normal);
    }
    normalize(v);
    return v;
}

vec3 oct_byte_to_vec(uint byte){ // 204 102 240
    return vec3(float(bool(byte & 102)), float(bool(byte & 204)), float(bool(byte & 240))) - vec3(0.5);
}

vec3 biggest_axis(vec3 v){
    float x = float(uint(abs(v.x) > abs(v.y) && abs(v.x) > abs(v.z)));
    float y = float(uint(abs(v.y) > abs(v.x) && abs(v.y) > abs(v.z)));
    float z = float(uint(abs(v.z) > abs(v.x) && abs(v.z) > abs(v.y)));

    return v * vec3(x,y,z);
}

uint get_byte(uint src, uint byte){
    src = src << (3 - byte) * 8;
    src = src >> 3 * 8;
    return src;
    src = src << 3 * 8;
    return src;
}

uint read_vbuffer(uint address){
    uint vecadr = uint((address - mod(address, 4)) / 4);
    vec4 vec = data[vecadr];

    return floatBitsToUint(vec[uint(mod(address, 4))]);
}

const float EPSILON = 0.000002;


bool inbox(vec3 pos, float size){
    return (pos.x + EPSILON >= size * -0.5) && (pos.x - EPSILON <= size * 0.5) && (pos.y + EPSILON >= size * -0.5) && (pos.y - EPSILON <= size * 0.5) && (pos.z + EPSILON >= size * -0.5) && (pos.z - EPSILON <= size * 0.5);
}

struct RayTarget{
    vec3 hitlocation;
    vec3 nodelocation;
    vec3 hitnormal;
    vec3 raydir;
    float t;
    float specular;
    uint node;
};

RayTarget cast_ray_v_box(vec3 start, vec3 dir, float size, vec3 nodelocation, uint node, vec3 globalray, float specular) {
    float t1 = (size * -0.5 - start.x) / dir.x;
    float t2 = (size * 0.5 - start.x) / dir.x;

    float t3 = (size * -0.5 - start.y) / dir.y;
    float t4 = (size * 0.5 - start.y) / dir.y;

    float t5 = (size * -0.5 - start.z) / dir.z;
    float t6 = (size * 0.5 - start.z) / dir.z;

    vec3 pos1 = start + dir * t1;
    vec3 pos2 = start + dir * t2;
    vec3 pos3 = start + dir * t3;
    vec3 pos4 = start + dir * t4;
    vec3 pos5 = start + dir * t5;
    vec3 pos6 = start + dir * t6;

    float b1 = float(inbox(pos1, size)) * t1;
    float b2 = float(inbox(pos2, size)) * t2;
    float b3 = float(inbox(pos3, size)) * t3;
    float b4 = float(inbox(pos4, size)) * t4;
    float b5 = float(inbox(pos5, size)) * t5;
    float b6 = float(inbox(pos6, size)) * t6;

    if(b1 <= 0.0){ b1 = FLT_MAX; }
    if(b2 <= 0.0){ b2 = FLT_MAX; }
    if(b3 <= 0.0){ b3 = FLT_MAX; }
    if(b4 <= 0.0){ b4 = FLT_MAX; }
    if(b5 <= 0.0){ b5 = FLT_MAX; }
    if(b6 <= 0.0){ b6 = FLT_MAX; }

    float min =min(b1, min(b2, min(b3, min(b4, min(b5, b6)))));

    if(abs(b1 - min) < EPSILON){return RayTarget(nodelocation + pos1, nodelocation, biggest_axis(pos1), globalray, b1, specular, node);}
    if(abs(b2 - min) < EPSILON){return RayTarget(nodelocation + pos2, nodelocation, biggest_axis(pos2), globalray, b2, specular, node);}
    if(abs(b3 - min) < EPSILON){return RayTarget(nodelocation + pos3, nodelocation, biggest_axis(pos3), globalray, b3, specular, node);}
    if(abs(b4 - min) < EPSILON){return RayTarget(nodelocation + pos4, nodelocation, biggest_axis(pos4), globalray, b4, specular, node);}
    if(abs(b5 - min) < EPSILON){return RayTarget(nodelocation + pos5, nodelocation, biggest_axis(pos5), globalray, b5, specular, node);}
    if(abs(b6 - min) < EPSILON){return RayTarget(nodelocation + pos6, nodelocation, biggest_axis(pos6), globalray, b6, specular, node);}
}
/*
struct VoxelNode{
    childmask: u8,
    flags: u8,
    colour : [u8; 3],

    padding: [u8; 3],
    //8 to 64 more bytes of pointers... kinda 32 do
}
*/

uint get_voxel_childmask(uint ptr){
    return get_byte(read_vbuffer(ptr),0);
}
uint get_voxel_flags(uint ptr){
    return get_byte(read_vbuffer(ptr),1);
}
vec3 get_voxel_colour(uint ptr){
    return vec3(get_byte(read_vbuffer(ptr),2), get_byte(read_vbuffer(ptr),3) , get_byte(read_vbuffer(ptr + 1),0)) / 255.0;
}
float get_voxel_emission(uint ptr){
    return float(get_byte(read_vbuffer(ptr + 1),1) / 255.0);
}
float get_voxel_metalness(uint ptr){
    return float(get_byte(read_vbuffer(ptr + 1),2) / 255.0);
}
float get_voxel_roughness(uint ptr){
    return float(get_byte(read_vbuffer(ptr + 1),3)) / 255.0;
}
uint get_voxel_child(uint voxelptr, uint childindex){
    uint child = 1 << childindex;

    uint select = 1;
    uint ptr = voxelptr + 2;
    while (true) {
        if((select & child) != 0) {
            return read_vbuffer(ptr);
        }

        if((get_voxel_childmask(voxelptr) & select) != 0) {
            ptr++;
        }
        select = select << 1;
    }
}


RayTarget cast_ray_voxel(vec3 start, vec3 dir, uint root, float specular) {
    float size = 1.0;
    uint[8] activenodes;
    activenodes[0]=root;
    vec3[8] nodelocations;
    nodelocations[0]=vec3(0.0);

    RayTarget closestleaf = RayTarget(vec3(0.0),vec3(0.0),vec3(0.0),vec3(0.0),0.0,0.0,0);
    closestleaf.t = FLT_MAX;

    uint count = 1;
    while(count > 0) {

        RayTarget[8] targets;
        for(uint i=0; i < 8; i++) {
            targets[i].t = FLT_MAX;
        }

        for(uint i=0; i < count; i++) {
            targets[i]=cast_ray_v_box(start - nodelocations[i], dir, size, nodelocations[i], activenodes[i], dir, specular);
        }

        count=0;

        #define CMP(x, y) (targets[x].t > targets[y].t)
        #define SWAP(x, y) RayTarget tar = targets[x]; targets[x] = targets[y]; targets[y] = tar;
        #define CSWAP(x, y) if(CMP(x, y)){SWAP(x, y)}

/* sort
[[0,1],[2,3],[4,5],[6,7]]
[[0,2],[1,3],[4,6],[5,7]]
[[1,2],[5,6],[0,4],[3,7]]
[[1,5],[2,6]]
[[1,4],[3,6]]
[[2,4],[3,5]]
[[3,4]]
*/

        CSWAP(0,1) CSWAP(2,3) CSWAP(4,5) CSWAP(6,7)
        CSWAP(0,2) CSWAP(1,3) CSWAP(4,6) CSWAP(5,7)
        CSWAP(1,2) CSWAP(5,6) CSWAP(0,5) CSWAP(3,7)
        CSWAP(1,5) CSWAP(2,6)
        CSWAP(1,4) CSWAP(3,6)
        CSWAP(2,4) CSWAP(3,5)
        CSWAP(3,4)

        for(uint h = 0; h < targets.length(); h++){
        RayTarget hit = targets[h];
        if (hit.t < FLT_MAX) {
            if ((get_voxel_flags(hit.node) & 1) != 0 && hit.t < closestleaf.t) {
                closestleaf = hit;
            } else {
                uint newcount= count;
                uint[8] newnodes;
                vec3[8] newlocations;
                for(uint i=0; i < 8; i++) {
                    uint child = 1 << i;

                    if((get_voxel_childmask(hit.node) & child) != 0) {
                        newlocations[newcount] = + hit.nodelocation + (oct_byte_to_vec(child) * size / 2.0);
                        newnodes[newcount]= get_voxel_child(hit.node, i);
                        newcount++;
                    }
                }
                activenodes=newnodes;
                nodelocations= newlocations;
                count=newcount;
            }
        }}

        size /= 2.0;
    }

    return closestleaf;
}
const uint RPP = 20;
const uint RBC = 4; // + 1
const uint maxgroupsize = 32; // 2^(RBC - 1)
const float EMISSION = 15.0;
void main() {
    vec2 coords = vec2(gl_GlobalInvocationID.xy) / vec2(imageSize(img));
    vec2 pixelsize = vec2(1.0) / vec2(imageSize(img));

    vec3 colour = vec3(0.0);
    uint count = 0;



    RayTarget groups[RBC];

    for(uint r = 0; r < RPP; r++){
        vec2 randcoord = coords + (pixelsize * vec2(rand(vec2(coords) * float(r)),
 rand(vec2(coords) * float(r + 1))));
        vec3 raydir = ((randcoord.x * 2.0 - 1.0) * vec3(pc.right)) + ((randcoord.y * 2.0 - 1.0) * vec3(pc.up)) + vec3(pc.forward);
        vec3 start =  vec3(pc.campos);

        groups[0] = cast_ray_voxel(start, raydir, 1, 1);

        for(uint g = 1; g < RBC; g++){
            float specular= round(rand(randcoord));
            vec3 newdir=rand_dir_from_surf(groups[g-1].hitnormal, vec2(randcoord) * g);
            float roughness = get_voxel_roughness(groups[g-1].node);
            vec3 specdir=(roughness * newdir) + ((1.0 - roughness) * reflectvec(groups[g-1].raydir, groups[g-1].hitnormal));
            groups[g] = cast_ray_voxel(groups[g-1].hitlocation + (EPSILON * groups[g-1].hitnormal), (newdir * (1.0-specular)) + (specdir * specular), 1, specular);
        }

        colour += get_voxel_colour(groups[0].node);
        count++;
    }


    vec4 to_write = vec4(colour / count,1.0);

    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}"
        }
    }

    let shader = cs::Shader::load(device.clone()).expect("failed to create shader module");

    let compute_pipeline = Arc::new(
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let set0 = Arc::new(
        PersistentDescriptorSet::start(
            compute_pipeline
                .layout()
                .descriptor_set_layout(0)
                .unwrap()
                .clone(),
        )
        .add_image(image.clone())
        .unwrap()
        .build()
        .unwrap(),
    );
    let set1 = Arc::new(
        PersistentDescriptorSet::start(
            compute_pipeline
                .layout()
                .descriptor_set_layout(1)
                .unwrap()
                .clone(),
        )
        .add_buffer(vbuffer.clone())
        .unwrap()
        .build()
        .unwrap(),
    );

    let right = rotation
        * Vector3::<f64> {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
    let up = rotation
        * Vector3::<f64> {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
    let forward = rotation
        * Vector3::<f64> {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        };

    let push_constants = cs::ty::pushConstants {
        campos: [campos.x, campos.y, campos.z],
        right: [right.x, right.y, right.z],
        up: [up.x, up.y, up.z],
        forward: [forward.x, forward.y, forward.z],
        _dummy0: [0; 8],
        _dummy1: [0; 8],
        _dummy2: [0; 8],
    };

    let buf = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..width * height * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .dispatch(
            [(width / 8) as u32, (height / 8) as u32, 1],
            compute_pipeline.clone(),
            (set0.clone(), set1.clone()),
            push_constants,
        )
        .unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap()
        .build()
        .unwrap();

    let gpustart = std::time::Instant::now();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let gpuend = std::time::Instant::now();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width as u32, height as u32, &buffer_content[..]).unwrap();
    image.save("image2.png").unwrap();

    println!("Gpu took {} ms", (gpuend - gpustart).as_millis());

    return;

    println!("Hello voxels");

    let tree = allocate(1000);
    let node = tree.get_node(tree.allocate_node());
    node.put_child(0, 1);
    node.put_child(1, 2);
    node.put_child(7, 3);
    node.put_child(6, 4);

    let mut mats = Vec::new();
    for i in 0..8 {
        let child: u8 = 1 << i;
        if (child & (*node).childmask) != 0 {
            let xyz = oct_byte_to_vec(child);
            println!("{:?}", xyz);
            mats.push(Matrix4::from_translation(xyz));
        }
    }

    let (thread, win) = antelope::window::main_loop::<MeshRenderer, MeshFrame>();

    let meshinfo = MeshCreateInfo {
        verticies: vec![
            Vertex {
                position: [0.0, 0.0, 0.0],
                colour: [1.0, 0.0, 0.0],
                normal: [1.0, 0.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                texcoord: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 0.0, -0.0],
                colour: [0.0, 1.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                texcoord: [0.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                colour: [0.0, 0.0, 1.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                texcoord: [0.0, 0.0],
            },
            Vertex {
                position: [0.5, -0.0, -0.9],
                colour: [1.0, 1.0, 1.0],
                normal: [1.0, 1.0, 1.0],
                tangent: [1.0, 0.0, 0.0, 1.0],
                texcoord: [0.0, 0.0],
            },
        ],
        indicies: vec![0, 1, 2, 0, 2, 3, 0, 1, 3, 2, 1, 3],
    };

    let mesh = win.mesh_factory.create_mesh(meshinfo);

    let cam = RenderCamera {
        position: Vector3 {
            x: -6.0,
            y: 4.0,
            z: 5.0,
        },
        rotation: Quaternion::from(Euler::new(Deg(30.0), Deg(60.0), Deg(0.0))),
        aspect: 1.0,
        fov: 90.0,
        far: 10000.0,
        near: 0.1,
    };
    let mut angle: f64 = 0.0;

    let mut meshes = Vec::new();
    for x in &mats {
        meshes.push(mesh.clone());
    }
    loop {
        win.render_info.push(RenderInfo {
            meshes: meshes.clone(),
            mats: mats.clone(),
            camera: cam.clone(),
        });
        angle += 0.1;
        std::thread::sleep_ms(200);
    }

    thread.join().ok().unwrap();
}
