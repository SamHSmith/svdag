use antelope::camera::RenderCamera;
use antelope::cgmath::prelude::One;
use antelope::cgmath::{Deg, Euler, Matrix4, Quaternion, Vector3, Vector4};
use antelope::mesh::{Mesh, MeshCreateInfo, MeshFactory, PostVertex, RenderInfo, Vertex};
use antelope::window::{DemoTriangleRenderer, Frame, TriangleFrame, Window};
use antelope::{MeshFrame, MeshRenderer};
use lib::*;

const EPSILON: f64 = 0.0001;

fn cast_ray_v_box(start: Vector3<f64>, dir: Vector3<f64>, size: f64) -> f64 {
    let t1 = (size * -0.5 - start.x) / dir.x;
    let t2 = (size * 0.5 - start.x) / dir.x;

    let t3 = (size * -0.5 - start.y) / dir.y;
    let t4 = (size * 0.5 - start.y) / dir.y;

    let t5 = (size * -0.5 - start.z) / dir.z;
    let t6 = (size * 0.5 - start.z) / dir.z;

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

    let b1 = (1.0 - inbox(pos1.x, pos1.y, pos1.z, size) as i64 as f64) * std::f64::MAX;
    let b2 = (1.0 - inbox(pos2.x, pos2.y, pos2.z, size) as i64 as f64) * std::f64::MAX;
    let b3 = (1.0 - inbox(pos3.x, pos3.y, pos3.z, size) as i64 as f64) * std::f64::MAX;
    let b4 = (1.0 - inbox(pos4.x, pos4.y, pos4.z, size) as i64 as f64) * std::f64::MAX;
    let b5 = (1.0 - inbox(pos5.x, pos5.y, pos5.z, size) as i64 as f64) * std::f64::MAX;
    let b6 = (1.0 - inbox(pos6.x, pos6.y, pos6.z, size) as i64 as f64) * std::f64::MAX;

    (b1.max(t1)).min((b2.max(t2)).min((b3.max(t3)).min((b4.max(t4)).min((b5.max(t5)).min(b6.max(t6))))))
}

fn cast_ray_voxel(
    start: Vector3<f64>,
    dir: Vector3<f64>,
    root: &VoxelNode,
    tree: VoxelTree,
) -> [u8; 3] {
    let mut size: f64 = 1.0;
    let mut activenodes: Vec<&VoxelNode> = Vec::new();
    let mut nodelocations: Vec<Vector3<f64>> = Vec::new();
    activenodes.push(root);
    nodelocations.push(new_vec(0.0, 0.0, 0.0));

    let mut x = 0;
    while x < activenodes.len() {
        let mut hits: Vec<f64> = Vec::new();
        let mut targets: Vec<(f64, (&VoxelNode, Vector3<f64>))> = Vec::new();
        for i in 0..activenodes.len() {
            let hit = cast_ray_v_box(start - nodelocations[i], dir, size);
            hits.push(hit);
            targets.push((hit, (activenodes[i], nodelocations[i])));

            x += 1;
        }
        let hit = hits.iter().cloned().fold(0. / 0., f64::min);
        if (hit.is_normal() && hit < std::f64::MAX) {
            let mut hitnode = (targets[0].1).0;
            let mut hitlocation = (targets[0].1).1;
            for (h, (node, location)) in targets {
                if h == hit {
                    hitnode = node;
                    hitlocation = location;
                }
            }
            if hitnode.flags & 1 != 0 {
                //leaf
                return hitnode.colour;
            } else {
                for c in 0..8 {
                    let child: u8 = 1 << c;

                    if hitnode.childmask & child != 0 {
                        nodelocations.push(hitlocation + (oct_byte_to_vec(child) * size / 2.0));
                        activenodes.push(hitnode.get_child(tree, c));
                    }
                }
            }
        }

        activenodes.drain(0..x);
        nodelocations.drain(0..x);
        x = 0;
        size /= 2.0;
    }

    return [0, 0, 0];
}

fn new_vec(x: f64, y: f64, z: f64) -> Vector3<f64> {
    return Vector3 { x, y, z };
}

fn main() {

    let mut tree = allocate(500);
    let node: &mut VoxelNode = tree.allocate_and_get_node();
    node.put_child(1, tree.allocate_node());
    node.get_child(tree, 1).flags = 1;
    node.get_child(tree, 1).colour = [255, 50, 50];
    node.put_child(0, tree.allocate_node());
    let node2: &mut VoxelNode = node.get_child(tree, 0);
    node2.put_child(0, tree.allocate_node());
    node2.get_child(tree, 0).flags = 1;
    node2.get_child(tree, 0).colour = [230, 183, 0];
    node2.put_child(2, tree.allocate_node());
    node2.get_child(tree, 2).flags = 1;
    node2.get_child(tree, 2).colour = [0, 183, 235];
    node2.put_child(5, tree.allocate_node());
    node2.get_child(tree, 5).flags = 1;
    node2.get_child(tree, 5).colour = [200, 183, 235];
    node2.put_child(7, tree.allocate_node());
    node2.get_child(tree, 7).flags = 1;
    node2.get_child(tree, 7).colour = [100, 183, 235];

    //node.flags =1;
    node.colour=[255, 183, 235];

    let width: u32 = 1024;
    let height: u32 = 1024;

    let campos = new_vec(-0.2, -0.8, -1.8);
    let camrot = new_vec(-20.0, -0.0, 0.0);
    let rotation = Quaternion::from(Euler {
        x: Deg(camrot.x),
        y: Deg(camrot.y),
        z: Deg(camrot.z),
    });

    let cubepos = new_vec(0.0, 0.0, 0.0);

    let mut buffer: Vec<u8> = vec![0; (width * height * 3) as usize]; // Generate the image data;

    let startcpu = std::time::Instant::now();

    for x in 0..width {
        for y in 0..height {
            let fx = x as f64 / width as f64;
            let fy = y as f64 / height as f64;

            let index = ((y * width + x) * 3 as u32) as usize;
            let colour: [u8; 3] = cast_ray_voxel(
                campos - cubepos,
                rotation * new_vec((fx * 2.0 - 1.0), (fy * 2.0 - 1.0), 1.0),
                node,
                tree,
            );

            buffer[index] = colour[0];
            buffer[index + 1] = colour[1];
            buffer[index + 2] = colour[2];
        }
    }

    let cpuend = std::time::Instant::now();

    // Save the buffer as "image.png"
    image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, &buffer[..])
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
        Dimensions::Dim2d {
            width,
            height,
        },
        Format::R8G8B8A8Unorm,
        Some(queue.family()),
    )
        .unwrap();


    let vbuffer = unsafe {CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, (0..500).into_iter().map(|i| {
        *(tree.base as *const VoxelNode as *const u32).offset(i)
    })).unwrap()};

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
#version 450
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38
#define DBL_MAX 1.7976931348623158e+308
#define DBL_MIN 2.2250738585072014e-308

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
layout(set = 1, binding = 0) buffer VoxelBuffer {
    vec4[] data;
};

layout(push_constant) uniform pushConstants {
    dvec3 campos;
    dvec3 right;
    dvec3 up;
    dvec3 forward;
} pc;

dvec3 oct_byte_to_vec(uint byte){ // 204 102 240
    return dvec3(double(bool(byte & 102)), double(bool(byte & 204)), double(bool(byte & 240))) - dvec3(0.5);
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

const double EPSILON = 0.0001;


bool inbox(dvec3 pos, double size){
    return (pos.x + EPSILON >= size * -0.5) && (pos.x - EPSILON <= size * 0.5) && (pos.y + EPSILON >= size * -0.5) && (pos.y - EPSILON <= size * 0.5) && (pos.z + EPSILON >= size * -0.5) && (pos.z - EPSILON <= size * 0.5);
}

double cast_ray_v_box(dvec3 start, dvec3 dir, double size) {
    double t1 = (size * -0.5 - start.x) / dir.x;
    double t2 = (size * 0.5 - start.x) / dir.x;

    double t3 = (size * -0.5 - start.y) / dir.y;
    double t4 = (size * 0.5 - start.y) / dir.y;

    double t5 = (size * -0.5 - start.z) / dir.z;
    double t6 = (size * 0.5 - start.z) / dir.z;

    dvec3 pos1 = start + dir * t1;
    dvec3 pos2 = start + dir * t2;
    dvec3 pos3 = start + dir * t3;
    dvec3 pos4 = start + dir * t4;
    dvec3 pos5 = start + dir * t5;
    dvec3 pos6 = start + dir * t6;

    double b1 = double(inbox(pos1, size)) * t1;
    double b2 = double(inbox(pos2, size)) * t2;
    double b3 = double(inbox(pos3, size)) * t3;
    double b4 = double(inbox(pos4, size)) * t4;
    double b5 = double(inbox(pos5, size)) * t5;
    double b6 = double(inbox(pos6, size)) * t6;

    if(b1 == 0.0){ b1 = DBL_MAX; }
    if(b2 == 0.0){ b2 = DBL_MAX; }
    if(b3 == 0.0){ b3 = DBL_MAX; }
    if(b4 == 0.0){ b4 = DBL_MAX; }
    if(b5 == 0.0){ b5 = DBL_MAX; }
    if(b6 == 0.0){ b6 = DBL_MAX; }


    return min(b1, min(b2, min(b3, min(b4, min(b5, b6)))));
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

struct RayTarget{
    dvec3 hitlocation;
    double t;
    uint node;
};

struct LeafHit{
    double t;
    vec3 colour;
};

vec3 cast_ray_voxel(dvec3 start, dvec3 dir, uint root) {
    double size = 1.0;
    uint[8] activenodes;
    activenodes[0]=root;
    dvec3[8] nodelocations;
    nodelocations[0]=dvec3(0.0);

    LeafHit closestleaf = LeafHit(DBL_MAX, vec3(0.0));

    uint count = 1;
    while(count > 0) {

        RayTarget[8] targets;
        for(uint i=0; i < 8; i++) {
            targets[i].t = DBL_MAX;
        }

        for(uint i=0; i < count; i++) {
            targets[i]=RayTarget(nodelocations[i], cast_ray_v_box(start - nodelocations[i], dir, size), activenodes[i]);
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
        if (hit.t < DBL_MAX) {
            if ((get_voxel_flags(hit.node) & 1) != 0 && hit.t < closestleaf.t ) {
                closestleaf = LeafHit(hit.t, get_voxel_colour(hit.node));
            } else {
                uint newcount= count;
                uint[8] newnodes;
                dvec3[8] newlocations;
                for(uint i=0; i < 8; i++) {
                    uint child = 1 << i;

                    if((get_voxel_childmask(hit.node) & child) != 0) {
                        newlocations[newcount] = + hit.hitlocation + (oct_byte_to_vec(child) * size / 2.0);
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

    return closestleaf.colour;
}

void main() {
    dvec2 coords = dvec2(gl_GlobalInvocationID.xy) / dvec2(imageSize(img));

    dvec3 start =  pc.campos;
    dvec3 ray =  ((coords.x * 2.0 - 1.0) * pc.right) + ((coords.y * 2.0 - 1.0) * pc.up) + pc.forward;

/// cyan vec3(0.0, 183.0 / 255.0, 235.0 / 255.0)
    vec4 to_write = vec4(0.0,0.0,0.0,1.0);


    to_write = vec4(cast_ray_voxel(start, ray, 1), 1.0);

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

    let gpustart = std::time::Instant::now();

    let command_buffer = AutoCommandBufferBuilder::new(device.clone(), queue.family())
        .unwrap()
        .dispatch(
            [width / 8, height / 8, 1],
            compute_pipeline.clone(),
            (set0.clone(), set1.clone()),
            push_constants,
        )
        .unwrap()
        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap()
        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let gpuend = std::time::Instant::now();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, &buffer_content[..]).unwrap();
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
